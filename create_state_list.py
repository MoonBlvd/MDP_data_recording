
file = open('state_list.txt','w')
for a in range(3):
    for b in range(3):
        for c in range(3):
            for d in range(3):
                for e in range(2):
                    for f in range(2):
                        for g in range(2):
                            file.write("%s\n" % (str(g)+str(f)+str(e)+str(d)+str(c)+str(b)+str(a)))
