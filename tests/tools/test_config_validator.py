from tools.validate_config import validate_ast
import ast
import pytest
_TestConfig1 = '\n@config\nclass _TestConfig1:\n    pass\n'
_TestConfig2 = '\n@config\n@dataclass\nclass _TestConfig2:\n    a: int\n    """docstring"""\n'
_TestConfig3 = '\n@config\n@dataclass\nclass _TestConfig3:\n    a: int = 1\n'
_TestConfig4 = '\n@config\n@dataclass\nclass _TestConfig4:\n    a: Union[Literal[1], Literal[2]] = 1\n    """docstring"""\n'

@pytest.mark.parametrize(('test_config', 'expected_error'), [(_TestConfig1, 'must be a dataclass'), (_TestConfig2, 'must have a default'), (_TestConfig3, 'must have a docstring'), (_TestConfig4, 'must use a single Literal')])
def test_config(test_config, expected_error):
    tree = ast.parse(test_config)
    with pytest.raises(Exception, match=expected_error):
        validate_ast(tree)