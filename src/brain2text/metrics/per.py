def levenshtein(a, b):
    # a, b: lists of token ids
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[-1]

def per(pred, target):
    # pred/target: lists of token ids (phonemes)
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, target) / len(target)
