def randomconverter(cleanstrings, targets):

    # inputs are array of cleaned utterances and their target categories

    newutt = []
    newtarget= []
    for count in range(len(cleanstrings)):
        if (len(cleanstrings[count])>21):
            noofnew = 3  # number of new utterances to create from large strings
            noofelements = 20  # length of new randomly generated utterances

        else:
            #for shorter utterances, just randomize them without creating additional ones

            noofnew = 1
            noofelements = len(cleanstrings[count])

        for count2 in range(noofnew):
            newutt.append(''.join(random.sample((cleanstrings[count]), noofelements)))
            newtarget.append(targets[count])
    return (newutt, newtarget)
