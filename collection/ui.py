import numpy as np
from pynput import keyboard
from pynput import mouse
import datetime
import time
import os
import sys

# global constants
rawData = []
startTime = None
endTime = None
shiftModifier = False
numKeyPresses = 0
counter = 0
actualPassword = ".tie5Roanl"

def welcomeUser():
    print("Please enter your password (hint, it's \".tie5Roanl\" - for now!)")
    print("Press enter to submit your password entry.")

# intercept keyboard events using pynput handler, perform basic error 
def push_down(key):
    global startTime
    global rawData
    global shiftModifier
    global numKeyPresses
    global counter
    
    # potentially exit the listener
    if key == keyboard.Key.enter:
        endTime = time.time()
        numKeyPresses = 0
        return False
    
    # if alphanumeric, process it as such
    try:
        if startTime == None:
            startTime = time.time()
        rawData.append( (key.char, "DOWN", time.time() - startTime) )
        numKeyPresses += 1
        print("\r" + "*" * numKeyPresses, end ="")
    except AttributeError:
        if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
            shiftModifier = True

# asyncronous logic for when keys are released, with basic error checking
def release(key):
    global startTime
    global rawData
    global shiftModifier
    
    try:
        if shiftModifier:
            rawData.append( (rawData[-1][0], "UP", time.time() - startTime) )
            shiftModifier = False
        else:
            rawData.append( (key.char, "UP", time.time() - startTime) )
    except AttributeError:
        pass

# check if a given 'UP' entry has a corresponding down entry
def entryClosed(index, opener):
    global rawData
    for newIndex in range(index, len(rawData)):
        potentialCloser = rawData[newIndex]
        if potentialCloser[1] == "DOWN" and potentialCloser[0] == opener[0]:
            return True
    return False

# ensure that every up entry is 'closed' - i.e. there is a down event
def ensureCompleted():
    global endTime
    global startTime
    global rawData
    for index, opener in enumerate(rawData):

        # if this is already a closing entry, ignore it
        if opener[1] == "UP": continue
        
        # otherwise, check it's closed.  if not, close it at the end time
        if not entryClosed(index, opener):
            rawData.append((opener[0], "UP", endTime - startTime))

# find a given up entry's corresponding down entry, if it exists
def findPrevious(key):
    global rawData
    first = True
    for entry in rawData[::-1]:
        if first:
            first = False
            continue
        if entry[0] == key and entry[1] == "DOWN": return entry
        if entry[0] == key and entry[1] == "UP": return None
    return None

# find a given entry's corresponding up/down entry with bounded linear search from a given index
def findPreviousFromIndex(key, index):
    global rawData
    first = True
    index -= 1
    while index >= 0:
        entry = rawData[index]
        if entry[0] == key and entry[1] == "DOWN": return entry
        if entry[0] == key and entry[1] == "UP": return None
        index -=1
    return None

# clear rogue 'up' entries with no down entries from the list (normally happens with weird shift key antics)
def clearRogueUps():
    global rawData
    if rawData[-1][1] == "UP":
        data = findPrevious(rawData[-1][0])
        if data == None or data[1] == "UP":
            del rawData[-1]
    
    index = 0
    while True:
        if index == len(rawData): break
        entry = rawData[index]
        if entry[1] == "UP":
            data = findPreviousFromIndex(entry[0], index)
            if data == None or data[1] == "UP":
                del rawData[index]
                continue
        index += 1

# ensure that the password was properly entered
def passwordProperlyEntered():
    global rawData
    global actualPassword
    
    buildString = ""
    for entry in rawData:
        if entry[1] == "DOWN":
            buildString += entry[0]
    return buildString == actualPassword

# returns a single password attempt from user input
def getOnePassword():
    return welcomeUserAndCollectUserPasswordData(1, 0, verbose = False)

# actual harness function that gathers user password data entry attempts and returns them to the caller
# called from data.py
def welcomeUserAndCollectUserPasswordData(numPasswordsNeeded, numRunupNeeded, verbose = True):
    global rawData
    global endTime
    global startTime
    global shiftModifier
    global numKeyPresses
    global counter
    
    if verbose: welcomeUser()
    totalData = []

    i = 0
    while i < numPasswordsNeeded + numRunupNeeded:
        with keyboard.Listener(on_press=push_down, on_release=release) as listener:
            listener.join()
        
        # ensure that all entries in the data are closed
        ensureCompleted()
        clearRogueUps()
        
        # clear the global variables again
        startTime = None
        endTime = None
        shiftModifier = False
        numKeyPresses = 0
        counter = 0
        if passwordProperlyEntered():
            if i >= numRunupNeeded:
                totalData.append(rawData)
            if verbose: print("\nFantastic, now enter the password again!  (Trial {} of {}).".format(i + 1, numPasswordsNeeded + numRunupNeeded))
            i += 1
        else:
            print("\nPassword mis-entered.  Try again:")
        rawData = []

    if verbose: print("Great - we've finished gathering training data from you.  Please wait while we process this information")
    return totalData
