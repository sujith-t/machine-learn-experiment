library(forecast)

data(AirPassengers)
class(AirPassengers)

start(AirPassengers)

end(AirPassengers)

frequency(AirPassengers)

sum(is.na(tsdata))

summary(AirPassengers)

AirPassengers
########explore
tsdata<-ts(AirPassengers,frequency = 12)

ddata<-decompose(tsdata, "multiplicative")

plot(ddata)


plot(ddata$trend)


plot(ddata$seasonal)


plot(ddata$random)
########
plot(AirPassengers)
abline(reg=lm(AirPassengers~time(AirPassengers)))
cycle(AirPassengers)

#get a boxplot by cycle
boxplot(AirPassengers~cycle(AirPassengers, xlab="Date", ylab = "Passenger Numbers (1000's)" ,main ="Monthly Air Passengers Boxplot from 1949 to 1961"))
#stationarity
plot(AirPassengers)

#ask R for the best model
mymodel<-auto.arima(AirPassengers)
mymodel
#lets run with trace to compare the information criteria values
auto.arima(AirPassengers,ic="aic" ,trace = TRUE)

install.packages("tseries")
library(tseries)
adf.test(mymodel)

plot.ts(mymodel$residuals)


acf(ts(mymodel$residuals),main='ACF Residual')
pacf(ts(mymodel$residuals),main='PACF Residual')


#use the model to forecast for the next 10 years
myforecast<-forecast(mymodel,level=c(95),h=10*12)
plot(myforecast)


  Box.test(mymodel$resid, lag=5, type="Ljung-Box")
  Box.test(mymodel$resid, lag=10, type="Ljung-Box")
    Box.test(mymodel$resid, lag=15, type="Ljung-Box")