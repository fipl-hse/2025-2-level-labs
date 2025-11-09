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
    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self._balance = balance
    
    def deposit(self, amount: float) -> None:
        """Adds money to the account."""
        if self.validate_transaction(amount):
            self._balance += amount
    
    def withdraw(self, amount: float) -> None:
        """Subtracts money from the account if sufficient funds exist."""
        if self.validate_transaction(amount) and self._balance >= amount:
            self._balance -= amount
    
    def get_balance(self) -> float:
        """Returns the current balance of the account."""
        return self._balance
    
    def update_balance(self, amount: float) -> None:
        """Updates the balance."""
        if self.validate_transaction(amount):
            self._balance = amount

    def validate_transaction(self, amount: float) -> bool:
        """Checks if a transaction amount is valid."""
        return isinstance(amount, (int, float)) and amount > 0


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
    def __init__ (self, name: str, accounts: dict, next_account_number: int):
            self._name = name
            self.accounts = accounts
            self.next_number = next_account_number

    def create_account(self, user_name: str):
        account = {user_name: BankAccount(user_name)}

    def generate_account_number(self):
        self.__next_account_number += 1

    def deposit_to_account(self, account_number, amount):
        self._accounts[account_number] += amount



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


def main() -> None:
    """
    Launch listing.
    """
    # Work here
    print("Created classes")
    account = BankAccount('Ksenia', 1000)
    print(account.get_balance())
    account.withdraw(500)
    print(account.get_balance())
    account.deposit(300)
    print(account.get_balance())
    print(account._balance)
    account._validate_trasaction(-700)
    print(account.get_balance())


if __name__ == "__main__":
    main()
