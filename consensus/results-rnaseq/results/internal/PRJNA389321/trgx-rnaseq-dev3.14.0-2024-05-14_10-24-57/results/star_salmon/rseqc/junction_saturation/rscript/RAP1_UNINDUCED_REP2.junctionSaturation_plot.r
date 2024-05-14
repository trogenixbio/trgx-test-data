pdf('RAP1_UNINDUCED_REP2.junctionSaturation_plot.pdf')
x=c(5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100)
y=c(1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
z=c(4,8,11,15,18,23,26,30,33,35,37,42,46,48,48,53,54,57,62,63)
w=c(3,7,9,13,16,21,23,27,30,32,34,39,43,45,45,50,51,54,59,60)
m=max(0,0,0)
n=min(0,0,0)
plot(x,z/1000,xlab='percent of total reads',ylab='Number of splicing junctions (x1000)',type='o',col='blue',ylim=c(n,m))
points(x,y/1000,type='o',col='red')
points(x,w/1000,type='o',col='green')
legend(5,0, legend=c("All junctions","known junctions", "novel junctions"),col=c("blue","red","green"),lwd=1,pch=1)
dev.off()