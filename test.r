library(ggplot2)
data(iris)
iris_scaled <- scale(iris[,1:4])
set.seed(123)
kmeans_result <- kmeans(iris_scaled, centers = 3)
iris$Cluster <- as.factor(kmeans_result$cluster)
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Cluster)) + 
  geom_point() + theme_minimal() + labs(title = "K-Means Clustering of Iris Data")