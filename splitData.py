import os
import random


if __name__ =="__main__":

    files = os.listdir("/data/shw/zhixing/ZH001/Annotations")

    sets = []
    for f in files:
        sets.append(f.split(".")[0])

    train_f = open("/data/shw/zhixing/ZH001/ImageSets/train.txt","w")
    test_f = open("/data/shw/zhixing/ZH001/ImageSets/test.txt", "w")

    random.shuffle(sets)
    print(len(sets))
    print(len(sets[:250]))
    print(len(sets[250:]))

    for s in sets[:250]:
        train_f.write(s+"\n")
    train_f.close()

    for s in sets[250:]:
        test_f.write(s+"\n")
    test_f.close()