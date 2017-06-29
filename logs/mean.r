library(ggplot2)
require(reshape2)


input="/scratch/tcriscuolo/github/UClass/RBM/logs/cd/cd_1/pseudo.txt"

nbatchs = 20 + 1
ntestes = 30

data = read.table(input, header = F)
data = matrix(data$V1, ncol = ntestes ,nrow = nbatchs ,byrow = F)

cd = apply(data, 1, mean)[-1]
cd2 = 2*y
x = 1:(nbatchs - 1)


df = data.frame(x, cd, cd2)
melted = melt(df, id.vars = 'x')

p = ggplot(melted, aes(x = x, y = value,
                       color = variable))

plot = p + geom_point() +
      labs(x = "Epoch", y = "Estimated pseudo likelihood") +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
      panel.background = element_blank(), axis.line = element_line(colour = "black"))
plot
