# Home Feed Algorithm Analysis: How One Account Is Affected

## Overview

The home-mixer system is a multi-stage pipeline that determines what content appears on a user's For You timeline. Each account (viewer) is affected by a complex interplay of retrieval, scoring, filtering, and ranking stages. Below is a detailed breakdown of every factor that positively or negatively affects a single account's experience.

---

## 1. POST TYPE (Positive & Negative Effects)

### Original Tweets (non-reply, non-retweet)
- **Positive**: Eligible for bidirectional follow reply/dwell weight boost when author mutually follows you
- **Positive**: Full participation in all engagement weight signals (favorite, reply, retweet, share, quote, etc.)
- **Positive**: Eligible for post_unexplored scoring boost when in-network
- Source: anking_scorer.rs:180-184

### Replies (in_reply_to_tweet_id.is_some())
- **Negative**: NOT eligible for bidirectional follow reply weight boost
- **Negative**: Out-of-network replies receive OonWeightFactor discount (default 0.75x)
- **Neutral**: Reply weight (default 5.0) is one of the highest positive signals
- Source: param.rs:283, anking_scorer.rs:180-184, 744-754

### Retweets (etweeted_tweet_id.is_some())
- **Negative**: NOT eligible for bidirectional follow reply weight boost
- **Negative**: Out-of-network retweets receive OonWeightFactor discount
- **Neutral**: Retweet weight (default 1.0) is moderate
- Source: param.rs:296, anking_scorer.rs:180-184, 744-754

### Quotes (quoted_tweet_id.is_some())
- **Positive**: Quote weight (default 5.0) is one of the highest positive signals
- **Positive**: Quoted click and quoted VQV signals contribute additional score
- **Neutral**: No special discounts applied
- Source: param.rs:332, 337, 340

### Videos
- **Positive**: Video quality view (VQV) signal contributes to score (weight 0.05)
- **Positive**: Video open signal contributes (weight 0.05)
- **Negative**: Videos under MinVideoDurationMs (default 10,000ms) get VQV weight set to 0
- **Negative**: Immersive video quality views tracked separately
- Source: param.rs:304, 317, 682

### Photos
- **Positive**: Photo expand signal contributes (weight 0.05)
- Source: param.rs:299

### Media Posts (has_media)
- **Neutral**: Hydrated but not directly scored differently unless video duration filters apply

---

## 2. TIME OF POST

### Post Age
- **Negative**: Posts older than MAX_POST_AGE (48 hours) are filtered out
- **Negative**: Cold start eligibility limited by ColdStartMaxPostAgeSecs (default 86,400 = 24 hours)
- **Positive**: Newer posts within the scoring window are preferred
- Source: config.rs:50, param.rs:648

### Engagement Recency (Dwell Regret Gate)
- **Positive**: Recent engagement signals (within 28-day window) boost ranking
- **Negative**: days_since_last engagement has negative weight (-0.03) in gate model
- **Negative**: Inactivity penalizes: 
_1d weight (-0.24) means posts without recent engagement get lower scores
- **Negative**: 
_7d weight (-0.08) penalizes inactivity over 7 days
- Source: param.rs:538, alue_model_gate.rs:37-38

### Active Session
- **Positive**: Posts with recent impressions in 5-minute active window get ctive_secs_5m_residual_norm boost
- **Neutral**: UAS (User Action Sequence) window is 300,000ms (5 minutes)
- Source: param.rs:418, config.rs:52

---

## 3. FOLLOWERS

### Viewer Follower Count
- **Negative**: Higher follower count has slight negative weight (-0.07) in gate model
- **Neutral**: Follower count capped at 10,000,000 for scoring
- Source: param.rs:538, alue_model_gate.rs:45

### Author Follower Count
- **Positive**: Author follower count is sent to VM Ranker for ranking context
- **Positive**: Cold start eligibility limited by ColdStartFollowerCap (default 1,000 followers)
- **Neutral**: Author followers used for diversity calculations
- Source: m_ranker.rs:188, param.rs:640

### Following Count (Viewer)
- **Positive**: Following count is a feature in gate model (weight +0.06)
- **Positive**: More following = more in-network candidates retrieved
- **Negative**: New users need minimum NEW_USER_MIN_FOLLOWING (5) to qualify for OON weight discount
- Source: param.rs:52, 538, anking_scorer.rs:693

---

## 4. BLOCK LIST

### Blocked User IDs (Viewer's Block List)
- **Hard Filter**: AuthorSocialgraphFilter removes ALL posts from blocked users
- **Hard Filter**: Posts quoting blocked users are removed
- **Hard Filter**: Posts retweeting blocked users are removed
- Source: uthor_socialgraph_filter.rs:30, 36-43

### Author Blocks Viewer
- **Hard Filter**: Posts from users who block the viewer are removed
- **Hard Filter**: Posts from quoted authors who block the viewer are removed
- Source: uthor_socialgraph_filter.rs:31, 33-34

### Block Signal in Scoring
- **Catastrophic Negative**: BlockAuthorWeight = -31.2
- **Catastrophic Negative**: In dwell regret mode, 
eg_block_author = -8,000
- **Negative**: Block actions tracked as negative feedback in gate model (NEGFB_ACTIONS)
- Source: param.rs:434, 520, alue_model_gate.rs:78-85

---

## 5. MUTED USERS

### Muted User IDs
- **Hard Filter**: AuthorSocialgraphFilter removes ALL posts from muted users
- **Catastrophic Negative**: MuteAuthorWeight = -58.8 (even worse than block in scoring)
- **Catastrophic Negative**: In dwell regret mode, 
eg_mute_author = -15,000
- Source: uthor_socialgraph_filter.rs:29, param.rs:440, 526

### Muted Keywords
- **Hard Filter**: MutedKeywordFilter removes posts matching any muted keyword
- **Tokenization**: Keywords are tokenized and matched against tweet text
- **Case-insensitive**: Matching is case-insensitive
- **Phrase support**: Multi-word phrases are supported (e.g., "crypto scam")
- **Hashtag matching**: Keywords without # match hashtags too
- **Unicode support**: Handles accented characters and unicode
- Source: muted_keyword_filter.rs

---

## 6. ENGAGEMENT SIGNALS (Positive)

### Explicit Engagement Signals (what you actively do)
| Signal | Weight | Impact |
|--------|--------|--------|
| Share via Copy Link | 20.0 | HIGHEST positive signal |
| Reply | 5.0 | Very high |
| Quote | 5.0 | Very high |
| Share via DM | 5.0 | Very high |
| Follow Author | 4.0 | High |
| Share | 2.0 | High |
| Retweet | 1.0 | Moderate |
| Favorite | 0.5 | Moderate |
| Click | 0.4 | Moderate |
| Open Link | 0.2 | Low |
| Bookmark | 0.0 | Tracked but no weight |
- Source: param.rs:282-350

### Implicit Engagement Signals (passive behavior)
| Signal | Weight | Impact |
|--------|--------|--------|
| Photo Expand | 0.05 | Low positive |
| Video Quality View | 0.05 | Low positive |
| Video Open | 0.05 | Low positive |
| Dwell Time | 0.0 | Used in dwell regret |
| Click Dwell Time | 0.0 | Used in dwell regret gate |
- Source: param.rs:297-331, implicit_engagement_signals_query_hydrator.rs

### Bidirectional Follow Boost
- **Positive**: If you and the author mutually follow each other, reply weight gets +15.0 boost
- **Positive**: Dwell weight also eligible for boost (currently 0.0)
- **Condition**: Only applies to original posts (not replies or retweets)
- Source: param.rs:284-295, anking_scorer.rs:180-184

### Post Unexplored Signal
- **Positive**: Posts you have not yet seen/engaged with get boost (weight 0.02)
- **Multiplicative Mode**: Can multiply dwell_time weight instead of additive
- **In-Network Only**: By default only applies to in-network posts
- Source: param.rs:351-374

---

## 7. ENGAGEMENT SIGNALS (Negative)

| Signal | Weight | Dwell Regret Weight | Impact |
|--------|--------|-------------------|--------|
| Report | -234.0 | -60,000 | MOST negative signal |
| Mute Author | -58.8 | -15,000 | Catastrophic |
| Not Interested | -43.2 | -10,000 | Catastrophic |
| Block Author | -31.2 | -8,000 | Catastrophic |
| Not Dwelled | -0.02 | N/A | Very small penalty |

### Tracked Negative Feedback Actions
- ClientTweetNotInterestedIn
- ClientTweetSeeFewer
- ClientTweetNotRelevant
- ClientTweetReport
- ClientTweetBlockAuthor
- ClientTweetMuteAuthor
- Source: alue_model_gate.rs:78-85

---

## 8. NETWORK POSITION (In-Network vs Out-of-Network)

### In-Network (authors you follow)
- **Positive**: Full score applied (no discount)
- **Positive**: Eligible for post_unexplored boost (default in-network only)
- **Positive**: Eligible for bidirectional follow boost
- Source: anking_scorer.rs:744-754

### Out-of-Network (recommended content)
- **Negative**: OonWeightFactor discount applied (default 0.75x of original score)
- **Negative**: Topic-based OON uses lower TopicOonWeightFactor (default 0.5x)
- **Negative**: New users with few following get extreme discount (NEW_USER_OON_WEIGHT_FACTOR = 0.00001)
- **Negative**: In-network replies and retweets also get OON discount when EnableOonRescoreForInNetworkRepliesRetweets is true
- Source: anking_scorer.rs:681-700, param.rs:247-271

---

## 9. AUTHOR DIVERSITY

### Diversity Mechanism
- **Negative**: Posts from the same author are penalized using exponential decay
- **Decay Factor**: Default 0.5 (each subsequent post from same author gets 50% of previous score)
- **Floor**: Minimum 0.25x (posts from same author never drop below 25% of original score)
- **Slate Context**: Tracks how many posts from each author are in the candidates pool
- Source: anking_scorer.rs:614-616, 655-679, param.rs:222-239

---

## 10. CONTENT SAFETY FILTERS

### Visibility Filtering (VF)
- **Hard Filter**: Posts with SafetyResult action=Drop are removed
- **Hard Filter**: Any FilteredReason (AuthorBlockViewer, etc.) causes removal
- Source: f_filter.rs

### NSFW Content
- **Hard Filter**: Out-of-network NSFW authors from simclusters source are filtered
- **Negative**: NSFW authors get escalated to MediumRisk brand safety verdict
- Source: oon_nsfw_simclusters_filter.rs, ds_brand_safety_vf_hydrator.rs

### Brand Safety
- **Negative**: Posts with HighRisk or MediumRisk brand safety verdicts receive lower scores
- **Negative**: NSFW authors always escalate to MediumRisk
- **Negative**: Ancestors with risky labels escalate the overall verdict
- Source: ds_brand_safety_vf_hydrator.rs

---

## 11. CONTENT QUALITY FILTERS

### Age Filter
- **Hard Filter**: Posts older than configured age threshold are removed
- Source: ge_filter.rs

### Previously Seen Posts
- **Hard Filter**: Posts already seen by the viewer are deduplicated
- **Hard Filter**: Posts from bloom filter entries (impression history) are filtered
- Source: previously_seen_posts_filter.rs, previously_served_posts_filter.rs

### Duplicate Filtering
- **Hard Filter**: Duplicate posts are removed
- **Hard Filter**: Duplicate conversation threads are deduplicated
- **Hard Filter**: Retweet deduplication prevents same content from multiple retweeters
- Source: drop_duplicates_filter.rs, dedup_conversation_filter.rs, etweet_deduplication_filter.rs

### Video Filter
- **Hard Filter**: Videos that do not meet duration requirements are filtered
- Source: ideo_filter.rs

### Result Size Filter
- **Hard Filter**: Limits total results to configured maximum (RESULT_SIZE = 35)
- Source: esult_size_filter.rs

---

## 12. COLD START MECHANISM

### New Author Boost
- **Positive**: Authors with fewer than ColdStartFollowerCap (1,000) followers get boosted
- **Positive**: Authors with fewer than ColdStartImpressionThreshold (1,000) impressions get boosted
- **Positive**: Boost slot is configurable between position 15-16 (ColdStartSlotMin/ColdStartSlotMax)
- **Positive**: Only posts within ColdStartMaxPostAgeSecs (24 hours) qualify
- Source: param.rs:621-656, uthor_cold_start.rs

### New Viewer Treatment
- **Positive**: New viewers (below NewUserAgeThresholdSecs) use different Phoenix retrieval cluster
- **Positive**: New users get special inference cluster for better recommendations
- Source: param.rs:183-219, phoenix_scorer.rs:28-42

---

## 13. USER PERSONALIZATION FEATURES

### Fetched About Viewer (Query Hydrators)
| Feature | Source | Effect |
|---------|--------|--------|
| Blocked user IDs | Social Graph | Hard filter |
| Muted user IDs | Social Graph | Hard filter |
| Muted keywords | User Settings | Hard filter |
| Followed user IDs | Social Graph | In-network candidates |
| Subscribed user IDs | Subscriptions | Subscription content |
| Follower count | Profile | Gate model feature |
| User demographics | Strato | Personalization |
| User inferred gender | Prediction | Personalization |
| User installed apps | App data | Personalization |
| Followed grok topics | Topic store | Topic-based retrieval |
| Followed starter packs | Starter pack store | Starter pack retrieval |
| IP location | GeoIP | Geo-based features |
- Source: query_hydrators/

### Fetched About Candidates (Candidate Hydrators)
| Feature | Source | Effect |
|---------|--------|--------|
| Author blocks viewer | Social Graph | Hard filter |
| Quoted author blocks viewer | Social Graph | Hard filter |
| Bidirectional follow status | Social Graph | Reply weight boost |
| Engagement counts | Counters | Ranking signal |
| Media info | Media store | Video duration filter |
| Safety labels | VF service | Brand safety |
| NSFW author flag | Clusters | NSFW filter |
| Tweet type metrics | Metrics | Type-specific scoring |
- Source: candidate_hydrators/

---

## 14. SCORING MODELS

### Phoenix Scorer (Neural Network)
- **Purpose**: Predicts engagement probabilities for each action type
- **Input**: User action sequence + candidate features
- **Output**: Scores for each engagement type (favorite, reply, retweet, etc.)
- **New User Handling**: Users below PhoenixRankerNewUserHistoryThreshold use separate cluster
- Source: phoenix_scorer.rs

### Value Model Ranker (VM Ranker)
- **Purpose**: Ranks candidates using a learned value model
- **DPP (Determinantal Point Process)**: Enforces diversity in ranking
- **Theta**: Controls diversity vs relevance tradeoff (default 0.65)
- **Max Selected Rank**: Limits candidates considered (default 150)
- Source: m_ranker.rs

### Value Model Gate (Dwell Regret)
- **Purpose**: Decides whether to use new scoring mode based on user behavior
- **Features**: 19 user behavior features (seq_len, n_fav, n_reply, n_rt_quote, etc.)
- **Threshold**: Users above threshold get new scoring; below get old scoring
- **Hysteresis**: Prevents oscillation between modes
- Source: alue_model_gate.rs

### Ranking Scorer (Weighted Linear)
- **Purpose**: Combines Phoenix scores with hand-tuned weights
- **Formula**: score = (pos - neg) / total_sum * NEGATIVE_SCORES_OFFSET
- **Negative Offset**: Small constant (0.001) ensures non-negative scores
- Source: anking_scorer.rs

---

## 15. SOURCE PIPELINE

### Candidate Sources
| Source | Max Results | Purpose |
|--------|-------------|---------|
| Phoenix | 1,000 | Primary neural retrieval |
| Thunder | 1,200 | Alternative retrieval |
| SimClusters | enabled | Interest-based retrieval |
| Phoenix MOE | 200 | Mixture of experts |
| Tweet Mixer | 800 | Mixed content |
| Who To Follow | position 6 | Follow suggestions |
| Ads | injected | Advertising |
| Cached Posts | 750 | Previously scored |
- Source: sources/, param.rs

### Retrieval Pipeline
1. **Retrieval**: Fetch candidates from multiple sources
2. **Filtering**: Apply hard filters (blocks, mutes, safety, duplicates)
3. **Hydration**: Enrich candidates with features
4. **Scoring**: Score using Phoenix neural network
5. **Ranking**: Re-rank using value model with diversity
6. **Selection**: Select top N for display
- Source: candidate_pipeline/

---

## 16. KEY ALGORITHMIC INSIGHTS

### Score Computation Formula
`
weighted_score = (favorite * 0.5) + (reply * 5.0) + (retweet * 1.0) + (quote * 5.0) 
               + (share * 2.0) + (share_via_dm * 5.0) + (share_via_copy_link * 20.0)
               + (follow_author * 4.0) + (click * 0.4) + (dwell_time * 0.004)
               - (not_interested * 43.2) - (block_author * 31.2) 
               - (mute_author * 58.8) - (report * 234.0)
`

### Dwell Regret Formula
`
positive = alpha_fav * (fav/mean_fav - 1) + alpha_reply * (reply/mean_reply - 1) + ...
negative = neg_not_interested * not_interested + neg_block * block + neg_mute * mute
modulation = 2 * sigmoid(positive/temperature) * exp(negative/temperature)
final_score = dwell_time * modulation
`

### Author Diversity Multiplier
`
multiplier = (1 - floor) * decay^k + floor
where k = number of posts from same author in pool
`

### Value Model Gate Score
`
gate_score = bias + sum(weight_i * log1p(feature_i))
serve_new_scoring = gate_score > threshold
`

---

## 17. SUMMARY: WHAT HELPS vs HURTS AN ACCOUNT

### What POSITIVELY Affects Visibility
1. **High engagement actions**: Share via copy link (20x), reply (5x), quote (5x), share via DM (5x)
2. **Mutual follows**: +15 boost on reply weight for original posts
3. **Fresh content**: Recent posts within 24-48 hours
4. **In-network position**: No OON discount
5. **New/unexplored content**: Small boost for novel posts
6. **Active session**: Recent engagement in 5-minute window
7. **Cold start**: New authors (< 1,000 followers) get boosted slots
8. **Content diversity**: Original posts preferred over replies/retweets

### What NEGATIVELY Affects Visibility
1. **Negative feedback**: Report (-234), Mute (-58.8), Block (-31.2), Not Interested (-43.2)
2. **Out-of-network**: 0.75x discount (or 0.00001x for new users)
3. **Author diversity**: Same-author posts decay by 50% each
4. **Old content**: Posts > 48 hours filtered; > 24 hours lose cold start
5. **Inactivity**: Negative gate weights for inactive users
6. **Safety filtering**: NSFW, brand safety, visibility filtering
7. **Block/mute lists**: Complete removal from feed
8. **Muted keywords**: Tokenized matching removes matching posts
9. **Short videos**: Under 10s get no VQV scoring
10. **Previously seen**: Bloom filter and deduplication remove repeats
