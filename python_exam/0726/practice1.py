# 1.다음에 나열한 것들 중에서 파이썬의 변수 이름으
# 로 사용할 수 없는 것들을 고르고,변수가 될 수
# 없는 이유를 쓰시오.
# ⑴ camel
# ⑵ False
# ⑶ false
# ⑷ break
# ⑸ money#
# ⑹ mail.com
# ⑺ 5var
# ⑻ variable_name
# ⑼ temp

#
# ⑵ False
# ⑷ break
# ⑸ money#
# ⑹ mail.com
# ⑺ 5var

# 2. 다음 문장의 괄호에 적절한 단어를 쓰시오.
# ⑴ 겹따옴표를 화면에 그대로 출력하려면 (     )을
# 사용한다.

# \"
# print("\"")

# ⑵ 여러 개의 자료들을 모아서 하나의 묶음으로 저장
# 하여 이용하는 자료를 (            )라고 한다.

# list
# ⑶ 함수 안에서 전역변수를 사용하려면 ()
# 키워드를 사용한다.

# global
# ⑷ 파이썬 코드를 저장한 파일의 확장자는()이며,
# 파일을 실행하려면 (             )을 사용한다.

# py, python
# ⑸ ()은 조건이 맞았을 때 묶어서 실행되는
# 코드로 들여쓰기로 만든다.

# if
# ⑹ 변수의 자료형을 확인하려면 (             )을
# 사용한다.

# type
# ⑺ 동일한 문자열을 여러 번 반복해 생성하여 하나의
# 문자열로 만들기 위해 (     )를 사용한다.

# *
# ⑻ 이미 만들어진 모듈을 이용하려면 파일의 첫 부분
# 에 (   )를 사용하여 이용하려는 모듈을
# 선언해주어야 한다.

# import
# ⑼ 함수의 매개변수가 기본값을 가질 수 있는데,이를
# (               )라고 한다.

# 디폴트 인수
#
# 3. 다음의 연산 결과를 쓰시오.
# ⑴ 2 > 1 + 1
# False
# ⑵ 3 > 1 and 3 < 5
# True
# ⑶ 6 % 2 == 0
# True
# ⑷ 7 // 3 == 2
# True

# 4.다음 코드에서 문제점이나 오류발생 부분을 표시
# 하고 오류 발생 이유와 해결 방법을 쓰시오.
# >>> a = 13
# >>> d = a + b + 1
# >>> print( "a = " + 3 )

# a = 13
# d = a + b + 1
# print("a = " + 3)

# b가 초기화가 안되있다.
# 5. 두 개의 실수를 입력받고, 연산자 + - * / 중 한
# 문자를 입력받아 연산한 결과를 출력하는 코드를
# 작성하시오.
# 조건1. 0으로 나누는 경우는 0으로 나눌 수 없음을
# 알려 준다.
# 조건2.조건2.+-*/이외의 문자를 입력하는 경우
# 연산자를 잘못 입력하였다고 알려준다.
# 조건3. 실행 결과
# 실수(a)를 입력하시오 : 10.5
# 실수(b)를 입력하시오 : 3.0
# 연산자를 입력하시오(+, -, *, /) : *
# 10.50 * 3.00 = 34.50
def q5():
    a = float(input("실수(a)를 입력하시오 : "))
    b = float(input("실수(b)를 입력하시오 : "))
    operator = input("연산자를 입력하시오(+, -, *, /) : ")
    if operator == "/" and b == 0:
        print("0으로는 나눌수 없습니다.")
    else:
        print(a, operator, b, "=", end=" ")
        if operator == '+':
            print(a + b)
        elif operator == '-':
            print(a - b)
        elif operator == '*':
            print(a * b)
        elif operator == '/':
            print(a / b)
        else:
            print("\n 연산자 입력이 잘못 되었습니다.")


# 6. 시간의 초(sec)를 정수로 입력받아 시, 분, 초로 계산하여 출력하는 코드를 작성하시오.
# 조건: 입력한 시간의 크기에 따라 ○ 시 ○ 분 ○ 초, ○ 분 ○ 초, 또는 ○ 초로 표시되도록 한다.
def q6():
    sec = int(input("초를 입력하세요 : "))
    # sec = 3700

    sec // 60 // 60
    sec % (60 * 60) // 60
    sec % (60 * 60) % 60
    if sec >= 3600:
        print(sec // 60 // 60, "시", sec % (60 * 60) // 60, "분", sec % (60 * 60) % 60, "초")
    elif sec >= 60:
        print(sec % (60 * 60) // 60, "분", sec % (60 * 60) % 60, "초")
    else:
        print(sec % (60 * 60) % 60, "초")


# 7. 무한 반복을 하면서 정수를 입력 받아 합을 계산하는 코드를 작성하시오.
# 조건1. 입력한 정수가 양수이면 더한다.
# 조건2. 입력한 정수가 음수이면 더하지 않는다.
# 조건3. 입력한 정수가 0이면 반복을 종료하고 계산된 합과 더한 숫자의 수를 출력한다.
def q7():
    total = 0
    while True:
        num = int(input("정수를 입력하세요 : "))

        if num < 0:
            break
        total += num
    print(total)


# 8. 정수를 입력받고, 입력된 정수 이하까지의 피보나치 수열을 출력하는 함수를 작성하시오.
# 조건 :피보나치 수열은 첫 번째 항은 0이고,두 번째 항은 1이며,이후에 이어지는 항들은 이전의 두 항을 더
# 한 값으로 이루어지는 수열임

temp = 87000
temp // 3600
