# final-research

深度聚类游戏分类推荐

In this study, we acquired 4000 game fragments, attempt to build one
Recommendation System to improve the accuracy of game recommendation.
The experiment tried to explore a potential method to measure the distance
between various articles under content-based recommendation.
The whole Recommendation System in this research base on the contentbased recommendation. Typically, the content-based recommendation has a
positive consequence. Moreover, it would not face the ”content-cold start”
problem. There are two sections in this recommendation structure. The
first data management section would use frames from games to trained one
deep learning neural network for visual representations extraction and build
recommendation database. Then aim to speed up the recommendation part,
the research would use a clustering method to manage data into a recommendation database. In the second recommendation section, the well-trained
model would use to extract the input of users which is the games playing by
user in real time. Then the recommendation system would calculate top-N
candidates for recommendation list according to the input of users.
