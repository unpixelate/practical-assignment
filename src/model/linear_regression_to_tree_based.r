library(data.table)
library(caTools)

setwd("C:/Users/Zheng_/Desktop/Prac Assessment/src/preprocess")

sti_dt = fread("combined_data.csv")
sum(sti_dt$Increase)
sti_dt$Date = NULL
sti_dt$Close = NULL
sti_dt$Volume_10year_treasury = NULL
sti_dt$close_snp = sti_dt$`Close_s&p`
sti_dt$`Close_s&p` = NULL
sti_dt$Volume_snp = sti_dt$`Volume_s&p`
sti_dt$`Close_s&p` = NULL

sti_dt$Increase = ifelse(sti_dt$Increase,1, 0)
names(sti_dt)

m1.close <- glm(Increase ~ Close_10year_treasury + close_snp + Close_copper + Close_gold + Close_hk_index + Value_us_sgd, family = binomial, data = sti_dt)
m1.full <- glm(Increase ~ Close_10year_treasury + close_snp + Close_copper + Close_gold + Close_hk_index + Value_us_sgd + Volume_copper + Volume_hk_index + Volume_oil ,family = binomial, data = sti_dt)
summary(m1.close)
summary(m1.full)

m4 <- step(m1.full)
fit2 <- lm(sti_dt$Increase ~ sti_dt$Close_hk_index, sti_dt)
m5 = step(fit2,direction="forward", trace = 1)
m6 <- glm(Increase ~ Close_copper +Volume_hk_index,family = binomial, data = sti_dt)
summary(m6)


library(tidyverse)      # data manipulation and visualization
library(lubridate)      # easily work with dates and times
library(fpp2)           # working with time series data
library(zoo)            # working with time series data

sti_dt$Date <- as.Date(sti_dt$Date, format="%d/%m/%Y")
savings <- sti_dt %>%
  select(Date, srate = Close)%>%
  mutate(srate_ma01 = rollmean(srate, k = 10, fill = NA),
         srate_ma02 = rollmean(srate, k = 25, fill = NA),
         srate_ma03 = rollmean(srate, k = 50, fill = NA),
         srate_ma05 = rollmean(srate, k = 100, fill = NA),
         srate_ma10 = rollmean(srate, k = 200, fill = NA))


savings %>%
  gather(metric, value, srate:srate_ma10) %>%
  ggplot(aes(Date, value, color = metric , group = 1)) +
  geom_line()
  