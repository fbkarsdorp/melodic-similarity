margin <- 1
dists <- seq(from=0, to=2, length.out=1000)
pos <- 0.25 * dists^2
neg_soft <- margin - dists
neg_soft[neg_soft < 0] = 0
neg_soft = neg_soft^2

margin <- 0.5
neg_hard <- (1-dists)^2 * (dists < margin)

pdf("../analyses/loss.pdf", width=6, height=4)
par(xpd=TRUE, mar=c(4, 4, 1, 1))
## par(xpd=FALSE)
red <- "#e74c3c"
blue <- "#3498db"
black <- "#34495e"
plot(dists, pos, type="l", col=black, ylab="loss", lwd=3, xlab="D", bty='L',
     cex.axis=1.3, cex.lab=1.3)
lines(dists, neg_soft, col=red, lwd=3)
lines(dists, neg_hard, col=blue, lwd=3)
legend(0.1, 1.1, legend=c("Positive loss", "Soft Negative Loss", "Hard Negative Loss"),
       col=c(black, red, blue), lwd=3, lty=1, cex=1.3, xpd=T, horiz=F, bty = "n")
dev.off()


