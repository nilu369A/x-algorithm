use super::builders::tweet_candidate;
use super::{Role, Row};
use crate::rules::fixtures::{allow, dropped};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration, TimelineHomeRecommendations};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    let trusted_friends = || {
        dropped(
            FilteredReason::UnspecifiedReason,
            "trusted_friends_tweet/drop/unspecified",
        )
    };
    vec![Row {
        name: "trusted_friends",
        post: tweet_candidate(|t| t.is_trusted_friends_tweet = true),
        expect: vec![
            (TimelineHomeHydration, Role::NonFollower, trusted_friends()),
            (TimelineHomeHydration, Role::Follower, trusted_friends()),
            (TimelineHomeHydration, Role::LoggedOut, trusted_friends()),
            (TimelineHomeHydration, Role::Author, allow()),
            (TimelineHome, Role::Follower, trusted_friends()),
            (TimelineHome, Role::Author, allow()),
            (
                TimelineHomeRecommendations,
                Role::NonFollower,
                trusted_friends(),
            ),
        ],
    }]
}
