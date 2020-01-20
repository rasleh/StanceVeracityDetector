#!/usr/bin/env bash

python -c 'import tweet_fetcher; from labeling import doccano_functions'

# Scrape twitter recent twitter data for 4 different hashtags, and add to database
#python -c 'tweet_fetcher.specific_search("#unpopularopinions AND -filter:retweets AND min_replies:5");
#tweet_fetcher.specific_search("#unpopularopinion AND -filter:retweets AND min_replies:5");
#tweet_fetcher.specific_search("#SorryNotSorry AND -filter:retweets AND min_replies:5");
#tweet_fetcher.specific_search("#changemymind AND -filter:retweets AND min_replies:5")'

# Generate stance and claim identification datasets, suitable for labeling using Doccano
python -c 'doccano_functions.generate_label_data("unlabeled.txt", "stance.jsonl", "claim.jsonl")'

