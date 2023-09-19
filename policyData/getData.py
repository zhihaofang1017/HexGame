import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import random
import torch


wd = webdriver.Chrome()
wd.implicitly_wait(10)

dir_path = os.path.dirname(os.path.realpath(__file__))
csvFile = open(os.path.join(dir_path, 'train_data.csv'), 'a')


def randomMoves():
    # Get random A to K
    random_letter = chr(random.randint(65, 75))
    # Get random 1 to 11
    random_number = random.randint(1, 11)
    return random_letter + str(random_number)


def initializeGame():
    # Random play for the 1 to 60 moves
    random_number = random.randint(0, 60)
    move = []
    for i in range(random_number):
        temp = randomMoves()
        while temp in move:
            temp = randomMoves()
        move.append(temp)
    # make move list to string
    move = ' '.join(move)
    # append the move to the textarea
    while True:
        try:
            wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[11]/td/table/tbody/tr[1]/td/textarea").send_keys(move)
            break
        except:
            time.sleep(0.1)
    # click apply button to take effect
    wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[11]/td/table/tbody/tr[2]/td/table/tbody/tr/td[2]/input").click()
    # Adjusting the computer Difficulty
    wd.find_element(By.XPATH, '/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[1]/td/table/tbody/tr[2]/td/table/tbody/tr[1]/td/input').click()
    wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[3]/td/table/tbody/tr[2]/td/table/tbody/tr[1]/td/input").click()
    wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[3]/td/table/tbody/tr[2]/td/table/tbody/tr[2]/td/input[3]").click()
    wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[1]/td/table/tbody/tr[2]/td/table/tbody/tr[2]/td/input[3]").click()


def writeData():
    initial_moves = wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[11]/td/table/tbody/tr[1]/td/textarea").get_attribute("value")
    wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[11]/td/table/tbody/tr[2]/td/table/tbody/tr/td[1]/input").click()
    steps = wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[11]/td/table/tbody/tr[1]/td/textarea").get_attribute("value")

    # change A1 to [0,0], A2 to [0,1],..., J10 to [9,9], J11 to [9,10]
    initial_moves = initial_moves.split()
    steps = steps.split()
    # split every move into two parts, the first part is first letter, the second part is the after part
    initial_moves = [tuple(i) for i in initial_moves]
    steps = [tuple(i) for i in steps]
    # change the part to number
    coordinates_ini = []
    coordinates_steps = []
    for i in range(len(initial_moves)):
        if len(initial_moves[i]) == 2:
            coordinates_ini.append([ord(initial_moves[i][0]) - 65, int(initial_moves[i][1]) - 1])
        else:
            if initial_moves[i][2] == '0':
                coordinates_ini.append([ord(initial_moves[i][0]) - 65, 9])
            else:
                coordinates_ini.append([ord(initial_moves[i][0]) - 65, 10])
    for i in range(len(steps)):
        if len(steps[i]) == 2:
            coordinates_steps.append([ord(steps[i][0]) - 65, int(steps[i][1]) - 1])
        else:
            if steps[i][2] == '0':
                coordinates_steps.append([ord(steps[i][0]) - 65, 9])
            else:
                coordinates_steps.append([ord(steps[i][0]) - 65, 10])
    # create a 11*11 board
    board = torch.zeros(11, 11, dtype=torch.int)
    color = 1

    # add initial moves to board
    for i in range(len(coordinates_ini)):
        board[coordinates_ini[i][0]][coordinates_ini[i][1]] = color
        color = -color

    # decide swap logic
    if coordinates_steps[0] == coordinates_steps[1]:
        coordinates_steps[1] = (-1, -1)

    # write step moves into csv file
    for i in range(len(coordinates_ini), len(coordinates_steps)):
        # board_to_go as training data
        board_to_go = board.clone()
        x, y = coordinates_steps[i][0], coordinates_steps[i][1]
        if color == 1:
            x, y = y, x
            board_to_go = board_to_go.transpose(0, 1)
        else:
            board_to_go = board_to_go * -1
        # write label and data info file
        row = [x, y] + board_to_go.reshape(121).tolist()
        row = ','.join([str(n) for n in row])
        csvFile.write(row + '\n')

        # update raw game board
        if x == -1:
            board[coordinates_steps[0][0]][coordinates_steps[0][1]] = 0
            board[coordinates_steps[0][1]][coordinates_steps[0][0]] = color
        else:
            board[coordinates_steps[i][0]][coordinates_steps[i][1]] = color
        color = -color
    csvFile.flush()


def main():
    while True:
        # Open the webpage
        wd.get('https://www.lutanho.net/play/hex.html')
        # Initialize the game
        initializeGame()

        # Get the sentence from the website
        sentence = wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[9]/td/table/tbody/tr[3]/td/input").get_attribute("value")
        while sentence != "Blue to move." or sentence != "Red to move.":
            if sentence == " Blue has won !" or sentence == " Red has won !":
                writeData()
                break
            sentence = wd.find_element(By.XPATH, "/html/body/div/form/table/tbody/tr/td[5]/table/tbody/tr[9]/td/table/tbody/tr[3]/td/input").get_attribute("value")


if __name__ == "__main__":
    main()
