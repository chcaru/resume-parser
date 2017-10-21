
Network

bufferLen = 4
s = bufferLen * 2 + 1
xT = Input((s,))
xP = Input((s,))
xTE = Embed()(xT)
x = concat([xTE, xP], axis=1)
r1 = BGRU(512, rs=true)(x)
yC = BGRU(512)(r1)