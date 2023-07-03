from ThesisPark import ThesisPark
import faulthandler
import os

if __name__ == '__main__':
    tesos=os.system('whoami')
    faulthandler.enable()
    initPark = ThesisPark()
    initPark.starter()

