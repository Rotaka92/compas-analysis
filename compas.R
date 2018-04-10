setwd("C:/Users/TapperR/Desktop/compas/compas-analysis")


library(dplyr)
library(ggplot2)
raw_data <- read.csv("compas-scores-two-years.csv")
nrow(raw_data)





df <- dplyr::select(raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A')
nrow(df)


#is there any correlation between the length of jail time and the decile score? yes, but little
df$length_of_stay <- as.numeric(as.Date(df$c_jail_out) - as.Date(df$c_jail_in))
cor(df$length_of_stay, df$decile_score)

#how is the distribution of the age look like
summary(df$age_cat)

#how is the distribution of the race look like
summary(df$race)



#how is the distribution of the score look like
summary(df$score_text)

#getting a crosstable for sex and race
xtabs(~ sex + race, data=df)

#how is the distribution of the sex look like
summary(df$sex)



#how many are recidivist
nrow(filter(df, two_year_recid == 1))
nrow(filter(df, two_year_recid == 1)) / nrow(df) * 100


#is there a downward trend in decile scores in the data, which contains only black/ white people (not that clear)

library(grid)
library(gridExtra)
pblack <- ggplot(data=filter(df, race =="African-American"), aes(ordered(decile_score))) + 
  geom_bar() + xlab("Decile Score") +
  ylim(0, 650) + ggtitle("Black Defendant's Decile Scores")
pwhite <- ggplot(data=filter(df, race =="Caucasian"), aes(ordered(decile_score))) + 
  geom_bar() + xlab("Decile Score") +
  ylim(0, 650) + ggtitle("White Defendant's Decile Scores")
grid.arrange(pblack, pwhite,  ncol = 2)


#crosstable over race and decile_score
xtabs(~ decile_score + race, data=df)




#creating some factors for some logistic regression
df <- mutate(df, crime_factor = factor(c_charge_degree)) %>%
  mutate(age_factor = as.factor(age_cat)) %>%
  within(age_factor <- relevel(age_factor, ref = 1)) %>%
  mutate(race_factor = factor(race)) %>%
  within(race_factor <- relevel(race_factor, ref = 3)) %>%
  mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%
  within(gender_factor <- relevel(gender_factor, ref = 2)) %>%
  mutate(score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore")))
model <- glm(score_factor ~ gender_factor + age_factor + race_factor +
               priors_count + crime_factor + two_year_recid, family="binomial", data=df)
summary(model)


df <- mutate(df, age_factor = as.factor(age_cat))
levels(df$age_factor)

df$age_factor <- relevel(df$age_factor, ref = 1)
levels(df$age_factor)

df <- mutate(df, race_factor = factor(race))
levels(df$race_factor)
df$race_factor <- relevel(df$race_factor, ref = 3)
levels(df$race_factor)



df <- mutate(df, score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore")))

df$score_factor


install.packages('ggfortify')
install.packages('rlang')

library(survival)
library(ggfortify)

data <- filter(filter(read.csv("cox-parsed.csv"), score_text != "N/A"), end > start) %>%
  mutate(race_factor = factor(race,
                              labels = c("African-American", 
                                         "Asian",
                                         "Caucasian", 
                                         "Hispanic", 
                                         "Native American",
                                         "Other"))) %>%
  within(race_factor <- relevel(race_factor, ref = 3)) %>%
  mutate(score_factor = factor(score_text)) %>%
  within(score_factor <- relevel(score_factor, ref=2))


levels(data$score_factor)

grp <- data[!duplicated(data$id),]
nrow(grp)


f <- Surv(start, end, event, type="counting") ~ score_factor
model <- coxph(f, data=data)
summary(model)




