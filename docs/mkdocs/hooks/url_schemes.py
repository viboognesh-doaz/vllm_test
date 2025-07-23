from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
import regex as re
'\nThis is basically a port of MyST parser’s external URL resolution mechanism\n(https://myst-parser.readthedocs.io/en/latest/syntax/cross-referencing.html#customising-external-url-resolution)\nto work with MkDocs.\n\nIt allows Markdown authors to use GitHub shorthand links like:\n\n  - [Text](gh-issue:123)\n  - <gh-pr:456>\n  - [File](gh-file:path/to/file.py#L10)\n\nThese are automatically rewritten into fully qualified GitHub URLs pointing to\nissues, pull requests, files, directories, or projects in the\n`vllm-project/vllm` repository.\n\nThe goal is to simplify cross-referencing common GitHub resources\nin project docs.\n'

def on_page_markdown(markdown: str, *, page: Page, config: MkDocsConfig, files: Files) -> str:
    """
    Custom MkDocs plugin hook to rewrite special GitHub reference links
    in Markdown.

    This function scans the given Markdown content for specially formatted
    GitHub shorthand links, such as:
      - `[Link text](gh-issue:123)`
      - `<gh-pr:456>`
    
    And rewrites them into fully-qualified GitHub URLs with GitHub icons:
      - `[:octicons-mark-github-16: Link text](https://github.com/vllm-project/vllm/issues/123)`
      - `[:octicons-mark-github-16: Pull Request #456](https://github.com/vllm-project/vllm/pull/456)`

    Supported shorthand types:
      - `gh-issue`
      - `gh-pr`
      - `gh-project`
      - `gh-dir`
      - `gh-file`

    Args:
        markdown (str): The raw Markdown content of the page.
        page (Page): The MkDocs page object being processed.
        config (MkDocsConfig): The MkDocs site configuration.
        files (Files): The collection of files in the MkDocs build.

    Returns:
        str: The updated Markdown content with GitHub shorthand links replaced.
    """
    gh_icon = ':octicons-mark-github-16:'
    gh_url = 'https://github.com'
    repo_url = f'{gh_url}/vllm-project/vllm'
    org_url = f'{gh_url}/orgs/vllm-project'
    urls = {'issue': f'{repo_url}/issues', 'pr': f'{repo_url}/pull', 'project': f'{org_url}/projects', 'dir': f'{repo_url}/tree/main', 'file': f'{repo_url}/blob/main'}
    titles = {'issue': 'Issue #', 'pr': 'Pull Request #', 'project': 'Project #', 'dir': '', 'file': ''}
    scheme = 'gh-(?P<type>.+?):(?P<path>.+?)(#(?P<fragment>.+?))?'
    inline_link = re.compile('\\[(?P<title>[^\\[]+?)\\]\\(' + scheme + '\\)')
    auto_link = re.compile(f'<{scheme}>')

    def replace_inline_link(match: re.Match) -> str:
        """
        Replaces a matched inline-style GitHub shorthand link
        with a full Markdown link.
        
        Example:
            [My issue](gh-issue:123) → [:octicons-mark-github-16: My issue](https://github.com/vllm-project/vllm/issues/123)
        """
        url = f"{urls[match.group('type')]}/{match.group('path')}"
        if (fragment := match.group('fragment')):
            url += f'#{fragment}'
        return f"[{gh_icon} {match.group('title')}]({url})"

    def replace_auto_link(match: re.Match) -> str:
        """
        Replaces a matched autolink-style GitHub shorthand
        with a full Markdown link.
        
        Example:
            <gh-pr:456> → [:octicons-mark-github-16: Pull Request #456](https://github.com/vllm-project/vllm/pull/456)
        """
        type = match.group('type')
        path = match.group('path')
        title = f'{titles[type]}{path}'
        url = f'{urls[type]}/{path}'
        if (fragment := match.group('fragment')):
            url += f'#{fragment}'
        return f'[{gh_icon} {title}]({url})'
    markdown = inline_link.sub(replace_inline_link, markdown)
    markdown = auto_link.sub(replace_auto_link, markdown)
    return markdown