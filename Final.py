
import Method1
import Method2
import Method3
import Comparison
import Recreate


def main():
    print('1-Clustering')
    print('2-Supervised Classifier')
    print('3-Dimension reduction before classifying')
    print('4-Comparison')
    print('5-Missing data')
    option = input('Enter the number of the preferred option')
    option = int(option)
    if option == 1:
        Method1.main()
    elif option == 2:
        Method2.main()
    elif option == 3 :
        Method3.main()
    elif option == 4 :
        Comparison.main()
    elif option == 5 :
        Recreate.main()
    else:
        print('wrong number')


main()







