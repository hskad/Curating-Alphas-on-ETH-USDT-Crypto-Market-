# Reinforcement Learning-Based Trading Strategy for Bitcoin

## Instructions

The project code file is named **`main.py`**. Ensure that **`BTC_2019_2023_1h.csv`** is saved in the same directory as the code file.

### Running the Code

1. Open **main_btc.py** and execute it.
2. Code has two funtions process_data(which preprocesses the data) and strat(creates signals).
3. The code will take some time to compute **fGBM signals**.
4. You’ll then create a new Q-table, which is required to train the model.
5. Continue executing the cells to train the Q-table based on the Reinforcement Learning actions defined in the code.

After training, the model is tested over the annual period of **2023**.

---
