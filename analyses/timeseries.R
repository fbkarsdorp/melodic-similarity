library(ggplot2)

df <- read.csv('../notebooks/timeseries.csv')
df <- df[df$year > 0,]

red <- "#e74c3c"
blue <- "#3498db"

pdf("../analyses/timeseries.pdf", height=3, width=10)
ggplot(df, aes(x=year, y=source, size = count, color=source, fill=source)) +
    geom_point(alpha=0.5, shape = 21) +
    theme_minimal() + xlab("Year") + ylab("Melody type") +
    ## scale_colour_discrete(guide = FALSE) +
    scale_fill_manual("Annual counts", values=c(red, blue), guide=F) +
    scale_colour_manual("Annual counts", values=c(red, blue), guide=F) + 
    scale_size_continuous(name="Annual counts",
                          breaks=c(10, 500, 1000, 2000, 4000),
                          range = c(1,20)) +
    theme(text = element_text(size=20),
          axis.text=element_text(size=20),
          legend.position="top") 
    ## scale_fill_discrete(guide=FALSE)
dev.off()
