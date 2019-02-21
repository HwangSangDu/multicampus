install.packages('tensorflow', dependencies = T)
library(tensorflow)
hello = tf$constant('hello')
print(hello)



sess = tf$Session()
print(sess$run(hello))


# sys.exit()
x = tf$Variable(3, name = "x")
y = tf$Variable(4, name = "y")


