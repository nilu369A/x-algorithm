use std::time::Duration;

use tonic::async_trait;
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_manhattan::s2s::S2sConfig;
use xai_manhattan::{NativeManhattanClient, Tenant};

use crate::util::popular_authors::{
    decode_stored, encode_stored, PopularAuthorsStore, StoredPopularAuthors,
};

const CLUSTER: &str = "omega";
const APP_ID: &str = "timelineservice_user_session_store";
const DATASET: &str = "tls_user_session_store";
const POPULAR_AUTHORS_PKEY: i64 = 0;
const POPULAR_AUTHORS_DATASET_ID: i32 = 1001;
const VERSION_NUM_V1: i32 = 1;
const TIMEOUT: Duration = Duration::from_millis(500);

fn tenant() -> Tenant {
    Tenant {
        cluster: CLUSTER.to_string(),
        app_id: APP_ID.to_string(),
        dataset: DATASET.to_string(),
    }
}

fn pkey() -> Vec<Vec<u8>> {
    vec![POPULAR_AUTHORS_PKEY.to_be_bytes().to_vec()]
}

fn lkey() -> Vec<Vec<u8>> {
    vec![
        POPULAR_AUTHORS_DATASET_ID.to_be_bytes().to_vec(),
        VERSION_NUM_V1.to_be_bytes().to_vec(),
    ]
}

pub struct ManhattanPopularAuthorsStore {
    client: NativeManhattanClient,
}

impl ManhattanPopularAuthorsStore {
    pub async fn new(dc: &str) -> anyhow::Result<Self> {
        let s2s = S2sConfig {
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            ca_cert_path: S2S_CHAIN_PATH.clone(),
        };
        let client = NativeManhattanClient::builder_from_tenant_s2s(&tenant(), dc, s2s)
            .timeout(TIMEOUT)
            .retries(1)
            .no_batch()
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create ManhattanPopularAuthorsStore: {e}"))?;
        Ok(Self { client })
    }
}

#[async_trait]
impl PopularAuthorsStore for ManhattanPopularAuthorsStore {
    async fn load(&self) -> Result<Option<StoredPopularAuthors>, String> {
        let item = self
            .client
            .get(tenant(), pkey(), lkey())
            .await
            .map_err(|e| format!("failed to get popular authors: {e}"))?;
        item.map(|it| decode_stored(it.value().as_bytes()))
            .transpose()
    }

    async fn save(&self, stored: &StoredPopularAuthors) -> Result<(), String> {
        self.client
            .put(tenant(), pkey(), lkey(), encode_stored(stored))
            .await
            .map_err(|e| format!("failed to put popular authors: {e}"))
    }
}
