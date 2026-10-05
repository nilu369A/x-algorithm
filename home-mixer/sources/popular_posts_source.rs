use crate::models::candidate::{PostCandidate, RetrievalSource};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePopularPostsSource, PopularPostsMaxPerAuthor, PopularPostsMaxRepliesRepostsPerAuthor,
    PopularPostsMaxResults, PopularPostsTopAuthors, ThunderAlgorithm, ThunderClusterId,
};
use crate::sources::thunder_source::THUNDER_CAPI_DECIDER;
use crate::util::popular_authors::{now_ms, PopularAuthorsCache};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::{
    ThunderCapiClient, ThunderClient, ThunderCluster,
};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_thunder_proto::in_network_posts_service_client::InNetworkPostsServiceClient;
use xai_thunder_proto::{GetInNetworkPostsRequest, LightPost, PerAuthorLimits};

const METRIC: &str = "PopularPostsSource";
const SERVED_TYPE: pb::ServedType = pb::ServedType::ForYouPhoenixRetrieval;

pub struct PopularPostsSource {
    pub thunder_client: Arc<ThunderClient>,
    pub thunder_capi_client: Option<Arc<dyn ThunderCapiClient + Send + Sync>>,
    pub popular_authors: Arc<PopularAuthorsCache>,
}

impl PopularPostsSource {
    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        request: GetInNetworkPostsRequest,
    ) -> Result<Vec<LightPost>, String> {
        let capi = self.thunder_capi_client.as_ref().filter(|_| {
            query
                .decider
                .as_ref()
                .is_some_and(|d| d.enabled(THUNDER_CAPI_DECIDER))
        });
        if let Some(capi) = capi {
            return Ok(capi
                .get_in_network_posts(request)
                .await
                .map_err(|e| format!("PopularPostsSource(capi): {e}"))?
                .posts);
        }
        let configured = ThunderCluster::parse(&query.params.get(ThunderClusterId));
        let cluster = ThunderCluster::resolve(configured, query.decider.as_ref());
        let channel = self
            .thunder_client
            .get_random_channel(cluster)
            .ok_or_else(|| "PopularPostsSource: no available channel".to_string())?;
        Ok(InNetworkPostsServiceClient::new(channel.clone())
            .get_in_network_posts(request)
            .await
            .map_err(|e| format!("PopularPostsSource: {e}"))?
            .into_inner()
            .posts)
    }
}

fn record(stage: &str, value: usize) {
    if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
        receiver.incr(METRIC, &[("stage", stage)], value as u64);
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PopularPostsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
            && !query.in_network_only
            && query.params.get(EnablePopularPostsSource)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let top_k = query.params.get(PopularPostsTopAuthors) as usize;
        self.popular_authors.maybe_spawn_refresh(now_ms());
        let authors = self.popular_authors.top_author_ids(top_k);
        record("requests", 1);
        record("authors", authors.len());
        if authors.is_empty() {
            record("no_authors", 1);
            return Ok(Vec::new());
        }

        let request = GetInNetworkPostsRequest {
            user_id: query.user_id,
            following_user_ids: authors,
            max_results: query.params.get(PopularPostsMaxResults),
            exclude_tweet_ids: query.seen_ids.to_vec(),
            algorithm: query.params.get(ThunderAlgorithm),
            debug: false,
            is_video_request: false,
            per_author_limits: Some(PerAuthorLimits {
                max_posts_per_author: Some(query.params.get(PopularPostsMaxPerAuthor)),
                max_replies_reposts_per_author: Some(
                    query.params.get(PopularPostsMaxRepliesRepostsPerAuthor),
                ),
            }),
        };
        let posts = self.fetch(query, request).await?;
        record("fetched", posts.len());

        Ok(posts
            .into_iter()
            .map(|p| PostCandidate {
                tweet_id: p.post_id as u64,
                author_id: p.author_id as u64,
                served_type: Some(SERVED_TYPE),
                retrieval_sources: vec![RetrievalSource::from_served_type(SERVED_TYPE)],
                ..Default::default()
            })
            .collect())
    }
}
