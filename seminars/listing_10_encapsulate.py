"""
Programming 2025.

Seminar 10.

Encapsulation.
"""

# pylint:disable=too-few-public-methods
from typing import Dict


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

    def __init__(self, owner: str, balance: float = 0.0) -> None:
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

    def __init__(self, name: str) -> None:
        self._name = name
        self._accounts: Dict[int, BankAccount] = {}
        self._next_account_number = 1

    def create_account(self, owner_name: str) -> int:
        """Creates a new bank account for a person."""
        account_number = self.generate_account_number()
        self._accounts[account_number] = BankAccount(owner_name)
        return account_number

    def generate_account_number(self) -> int:
        """Generates a new account number."""
        account_number = self._next_account_number
        self._next_account_number += 1
        return account_number

    def deposit_to_account(self, account_number: int, amount: float) -> None:
        """Deposits money into a specific account."""
        if account_number in self._accounts:
            self._accounts[account_number].deposit(amount)

    def provide_loan(self, account_number: int, amount: float) -> None:
        """Provides a loan to a specific account."""
        if account_number in self._accounts and amount > 0:
            self._accounts[account_number].deposit(amount)

    def get_account_balance(self, account_number: int) -> float:
        """Returns the balance of a specific account."""
        if account_number in self._accounts:
            return self._accounts[account_number].get_balance()
        return 0.0


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

    def __init__(self, name: str, account: BankAccount) -> None:
        self.name = name
        self.account = account

    def make_purchase(self, amount: float) -> None:
        """Subtracts money from the person's account for a purchase."""
        self.account.withdraw(amount)

    def transfer_money(self, recipient: "Person", amount: float) -> None:
        """Transfers money from this person's account to another person's account."""
        if self.account.get_balance() >= amount:
            self.account.withdraw(amount)
            recipient.account.deposit(amount)

    def add_funds(self, amount: float) -> None:
        """Adds money to the person's account."""
        self.account.deposit(amount)


def main() -> None:
    """
    Launch listing.
    """
    # Work here
    print("Created classes")
    account = BankAccount("Ksenia", 1000)
    print(account.get_balance())
    account.withdraw(500)
    print(account.get_balance())
    account.deposit(300)
    print(account.get_balance())
    print(account._balance)

    # Исправленный вызов метода
    is_valid = account.validate_transaction(-700)
    print(f"Is transaction valid: {is_valid}")
    print(account.get_balance())


if __name__ == "__main__":
    main()
