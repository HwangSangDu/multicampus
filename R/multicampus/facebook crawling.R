
fb_aouth = "EAAFh498V28EBAASFzDsD6gWylmBNsghFuxYB8zTXHgH8mxtJdTM8ZAti45Hu2eUttIMSZCVZCHhm8LmhAZC5vh1rZCK4W9UQLZCCkkC1yKj8Ru8BZCiSWr1xTdPqqCz4WKkLdnJaNA7cepAza4wdJAZCkUnbsk06dhEHdL5ZBqj4HCJTAGFNNjQmxZCeZC9ZBElLELojF5CJu42NNwZDZD"

pac <- c("Rfacebook", "base64enc", "ROAuth")

install.packages(pac)
library(Rfacebook)
library(base64enc)
library(ROAuth)


getUsers("me",token = fb_aouth,
         private_info = T)

getPage(
        "FacebookKorea",
        token=fb_aouth,
        n = 100)



data(mtcars)
mtcars
head(mtcars)
# quantile(as.data.frame(mtcars))
?attach

?adply
?apply



install.packages("plyr")
library(plyr)
?ddply





iris$Species
iris


?attach
llll <- list(xxx = iris , xxy = mtcars)
attach(llll)
xxx



