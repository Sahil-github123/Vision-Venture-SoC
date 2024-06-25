secret_no = 9
count = 0
while count < 3 :
    guess = int(input('Guess the number: '))
    count += 1
    if (guess == secret_no) :
        print('You Won!')
        break
    elif count < 31 :
        print('Try Again')
else :                                  ## Always executes unless the while loop is broken by a break statement. 
    print('You Lost!')                  ## In python, else exists for while loop also.