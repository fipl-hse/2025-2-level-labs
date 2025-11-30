"""
Programming 2025.

Seminar 10.

Encapsulation.
"""

# pylint:disable=too-few-public-methods


class BankAccount:
    """
    Represents a bank account.

    Instance attributes:
        owner (str): The owner of the account.
        balance (float): The current balance of the account.

    Instance methods:
        deposit(amount: float) -> None:
            Adds money to the account.
        withdraw(amount: float) -> None:
            Subtracts money from the account if sufficient funds exist.
        get_balance() -> float:
            Returns the current balance of the account.
        update_balance(amount: float) -> None:
            Updates the balance.
        validate_transaction(amount: float) -> bool:
            Checks if a transaction amount is valid.
    """
    def __init__(self, owner: str, balance: int = 0):
        self._owner = owner
        self._balance = balance

    def deposit(self, amount: float) -> None:
        self.update_balance(amount)
    
    def withdraw(self, amount: float) -> None:
        if self._validate_transaction(amount):
            self.update_balance(-amount)
    
    def update_balance(self, amount: float) -> None:
        self._balance += amount

    def _validate_transaction(self, amount: float)  -> bool:
        if self._balance < amount:
            return False
        return True
    def get_balance(self):
        return self._balance
    

class Bank:
    """
    Represents a bank that manages accounts and provides financial services.

    Instance attributes:
        name (str): The name of the bank.
        accounts (dict): A dictionary with BankAccount objects.
        next_account_number (int): The next account number to be assigned.

    Instance methods:
        create_account(owner_name: str) -> BankAccount:
            Creates a new bank account for a person.
        deposit_to_account(account_number: int, amount: float) -> None:
            Deposits money into a specific account.
        provide_loan(account_number: int, amount: float) -> None:
            Provides a loan to a specific account.
        get_account_balance(account_number: int) -> float:
            Returns the balance of a specific account.
        generate_account_number() -> int:
            Generates a new account number.
    """
    def __init__(self, name: str, accounts: dict, next_account_number: int):
        self.__name = name
        self._accounts = accounts
        self.__next_account_number = next_account_number

    def create_account(self, owner_name):
        a = BankAccount(owner_name)
        self._accounts[self._generate_account_number()] = BankAccount(owner_name)
        return a
    def provide_loan(self, account_number: int, amount: float) -> None:
        self._accounts[account_number].deposit(amount)

        
    def deposit_to_account(self, account_number: int, amount: float) -> None:
        self._accounts(account_number)

    def get_account_balance(self, account_number: int) -> float:
        return self._accounts[account_number].get_balance()
    


    def generate_account_number(self):
        self.__next_account_number += 1
        return self.__next_account_number


class Person:
    """
    Represents a person who can perform financial transactions through their bank account.

    Instance attributes:
        name (str): The name of the person.
        account (BankAccount): The person's bank account object.

    Instance methods:
        make_purchase(amount: float) -> None:
            Subtracts money from the person's account for a purchase.
        transfer_money(recipient: Person, amount: float) -> None:
            Transfers money from this person's account to another person's account.
        add_funds(amount: float) -> None:
            Adds money to the person's account.
    """
    def __init__(self, name: str, account):
        self.__name = name
        self.__account = account
    
    def make_purchase(self, amount: float) -> None:
        self.__account.withdraw(amount)

    def transfer_money(self, recipient: Person, amount: float) -> None:
        self.__account.withdraw(amount)
        recipient.add_fuds(amount)
        
    def add_funds(self, amount: float) -> None:
        self.__account.deposit(amount)

def main() -> None:
    """
    Launch listing.
    """
    # Work here
    print("Created classes")
    d = BankAccount("Teimur", 105000)
    #d.deposit(5000)
    #d.withdraw(5000)
    #print(d.get_balance())
    d.update_balance(1000)
    print(d.get_balance())
    print(dir(BankAccount))
    print(d._BankAccount_validate_transaction(500))
    print(d._validate_transaction(500))
    print(d._BankAccount_owner)

    


if __name__ == "__main__":
    main()
