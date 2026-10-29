import pytest

from app.security.policy import ActionClass, Risk, classify_command, classify_file_access


@pytest.mark.parametrize("argv, expected", [
    (["python", "-m", "pytest", "-q"], ActionClass.GREEN),
    (["npm", "test"], ActionClass.GREEN),
    (["git", "diff"], ActionClass.GREEN),
    (["ls"], ActionClass.GREEN),
    (["C:/Python/python.exe", "-m", "unittest"], ActionClass.GREEN),
    (["pip", "install", "requests"], ActionClass.AMBER),
    (["npm", "install", "left-pad"], ActionClass.AMBER),
    (["python", "-c", "print(1)"], ActionClass.AMBER),
    (["npx", "something"], ActionClass.AMBER),
    (["git", "commit", "-m", "x"], ActionClass.AMBER),
    (["mystery-binary"], ActionClass.AMBER),
    (["rm", "-rf", "build"], ActionClass.RED),
    (["del", "/s", "x"], ActionClass.RED),
    (["bash", "-c", "ls && rm -rf /"], ActionClass.RED),
    (["powershell", "-Command", "Remove-Item x"], ActionClass.RED),
    (["curl", "https://example.com"], ActionClass.RED),
    (["git", "push", "origin", "main"], ActionClass.RED),
    (["git", "reset", "--hard"], ActionClass.RED),
    (["npm", "publish"], ActionClass.RED),
    (["sudo", "ls"], ActionClass.RED),
    ([], ActionClass.RED),
])
def test_command_classes(argv, expected):
    assert classify_command(argv).action_class == expected


def test_file_access():
    assert classify_file_access("src/app.py", write=False).risk == Risk.SAFE
    assert classify_file_access("src/app.py", write=True).risk == Risk.MEDIUM
    assert classify_file_access(".env", write=False).risk == Risk.HIGH
    assert classify_file_access("config/prod.pem", write=False).risk == Risk.HIGH
    assert classify_file_access(".github/workflows/ci.yml", write=True).risk == Risk.HIGH
    assert classify_file_access(".git/config", write=True).risk == Risk.FORBIDDEN
