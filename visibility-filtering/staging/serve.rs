use crate::server::VFServer;
use crate::server_deps;
use crate::staging::reference;
use std::sync::Arc;
use xai_x_service_builder::{ServiceContext, XService};

pub struct StagingServer(Arc<VFServer>);

#[tonic::async_trait]
impl XService for StagingServer {
    type Config = ();

    async fn build(ctx: ServiceContext<()>) -> Self {
        let deps = server_deps::build(&ctx.datacenter).await;
        let comparator = reference::build(
            &ctx.datacenter,
            deps.init_deadline,
            &deps.filter_tweets,
            &deps.client_switches,
        )
        .await;
        Self(Arc::new(deps.into_server(comparator)))
    }

    fn register(self: Arc<Self>, routes: &mut tonic::service::RoutesBuilder) {
        Arc::clone(&self.0).register(routes);
    }
}
