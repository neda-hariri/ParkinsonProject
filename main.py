from ThesisPark import ThesisPark
import faulthandler

if __name__ == '__main__':
    faulthandler.enable()
    initPark = ThesisPark()
    initPark.starter()
