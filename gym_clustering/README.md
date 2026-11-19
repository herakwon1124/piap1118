# Gym member clustering

Who should get the monthly $10 extras credit?

`python3 analysis.py` refits k-means and rewrites the chart, the tables, and `slides.pdf`. Every figure on the slides comes from that run.

- Data: `gym_members.csv` (330 members, averages over the last 3 months).
- Split: 70% train / 30% test, `random_state=42` (231 / 99).
- Features: age, visits per month, minutes per visit, classes per month, months as a member, monthly extras spend. `member_id` is left out.
- Scaling: `StandardScaler` fit on the training members only.
- Model: k-means, k = 1 through 8. Chosen k = 4, where both the training and test spread flatten.
- Credit: a segment gets the $10 credit only when the member-level total for that segment is positive. That is the Newcomers segment.

Finance rules, applied one member at a time:

- extras spend under $20 and they receive the credit: +$5
- extras spend $20 or more and they receive the credit: −$10
- no credit: $0
