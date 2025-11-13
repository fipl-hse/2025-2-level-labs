"""
Generation by NGrams starter
"""

# pylint:disable=unused-import, unused-variable
from lab_3_generate_by_ngrams.main import TextProcessor

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    text_processor = TextProcessor("_")
    encoded = text_processor.encode(text)
    decoded = text_processor.decode(encoded)
    
    print("Кодирование и декодирование завершены успешно!")
    print(f"Исходный текст: {len(text)} символов")
    print(f"Закодированный текст: {len(encoded) if encoded else 0} токенов")
    print(f"Декодированный текст: {len(decoded) if decoded else 0} символов")
    print(f"Пример декодированного текста: {decoded[:100] if decoded else 'None'}...")

    result = decoded
    assert result is not None


if __name__ == "__main__":
    main()
