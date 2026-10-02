import json
import re
import html
import unicodedata
import urllib.parse
import urllib3
import requests
from requests.adapters import HTTPAdapter, Retry
import markdownify
from bs4 import BeautifulSoup
from .BaseDataExtractor import BaseDataExtractor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class _SmartMarkdownConverter(markdownify.MarkdownConverter):
    """
    Drop-in replacement for markdownify's default converter that detects the
    programming language of each <pre>/<code> block individually (from common
    class-naming conventions used by highlight.js, Prism, GFG, W3Schools,
    JavaTpoint, etc.) instead of forcing a single global language on every
    code block in the page.
    """

    LANG_PATTERN = re.compile(
        r"(?:language|lang|brush|highlight)[-:]([a-zA-Z0-9+#]+)", re.IGNORECASE
    )

    def _detect_language(self, el):
        classes = []
        code_el = el.find("code") if el.name == "pre" else el
        if code_el is not None:
            classes += code_el.get("class", []) or []
        classes += el.get("class", []) or []
        joined = " ".join(classes)
        match = self.LANG_PATTERN.search(joined)
        return match.group(1).lower() if match else ""

    def convert_pre(self, el, text, parent_tags):
        code_el = el.find("code") or el
        lang = self._detect_language(el)
        code_text = code_el.get_text()
        # Strip a single trailing newline so fences don't get extra blank lines
        code_text = code_text.rstrip("\n")
        return f"\n```{lang}\n{code_text}\n```\n"


class WebExtractor(BaseDataExtractor):
    """
    Universal web extractor with smart routing for:
      - Wikipedia (dedicated DOM cleaner: strips infoboxes, navboxes,
        citation markers, edit links, TOC, etc.)
      - News / blogs with JSON-LD (NewsArticle/Article/BlogPosting, incl. @graph)
      - Technical/tutorial sites (GFG, W3Schools, JavaTpoint, etc.) via DOM
        extraction with per-block code-language detection
    """

    def __init__(self, verify_ssl: bool = False, timeout: int = 15):
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        self.session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Domains we've already visited once this run — used to warm up a
        # session cookie on the site's homepage before hitting a deep page,
        # since some WAFs (Cloudflare/Akamai/PerimeterX-style) reject the
        # very first request from a new client but allow the second.
        self._warmed_domains = set()

        # Signatures that indicate we received an anti-bot / JS-challenge
        # page rather than real content. Checked only when extraction comes
        # back near-empty, purely for diagnostics.
        self.bot_block_patterns = re.compile(
            r"(?:enable javascript to continue|checking your browser|just a moment"
            r"|attention required|cf-browser-verification|cf-chl|captcha"
            r"|please verify you are (?:a )?human|access denied|request unsuccessful"
            r"|distil_r_captcha|perimeterx|px-captcha|ddos protection by|are you a robot)",
            re.IGNORECASE,
        )

        self.converter = _SmartMarkdownConverter(
            heading_style="ATX",
            strip=["a", "img"],
        )

        self.junk_tags = [
            "script", "style", "nav", "footer", "header", "aside", "form",
            "noscript", "iframe", "svg", "img", "canvas", "picture", "dialog",
            "button", "link",
        ]

        # Split into two groups because a single \b-bounded pattern misses
        # real-world CSS class names that concatenate words with no
        # separator (e.g. React/CSS-module classes like "WidgetComponent",
        # "TopBanner_abc123", "SidebarRail") — \bwidget\b never matches
        # inside "WidgetComponent" since there's no boundary before "idget".
        #
        # Short, generic tokens (nav, ad, cta, toc, menu) stay word-bounded
        # because without boundaries they'd false-positive constantly
        # (e.g. "ad" inside "head", "already", "load", "shadow").
        self.noise_word_boundary_patterns = re.compile(
            r"\b(?:nav|ad|ads|cta|toc|menu)\b", re.IGNORECASE
        )

        # Longer, distinctive keywords are unlikely to appear as accidental
        # substrings of unrelated words, so matching them anywhere in the
        # id/class string (no boundary required) catches concatenated names.
        self.noise_substring_patterns = re.compile(
            r"(?:navbar|header|footer|sidebar|side-bar|leftbar|rightbar|"
            r"left-bar|right-bar|ribbon|breadcrumb|comment|disqus|social|"
            r"share|related|recommend|trending|outbrain|taboola|advert|"
            r"sponsor|newsletter|subscribe|signup|sign-up|cookie|popup|"
            r"modal|banner|widget|promo|paywall|table-of-contents|masthead|"
            r"secondary|recirculation|mostpopular|most-popular|mostread|"
            r"most-read|topstories|top-stories|appbanner|app-banner|"
            r"installapp|install-app|downloadapp|download-app|stickycta|"
            r"sticky-cta|floatingcta|floating-cta|exitintent|exit-intent|"
            r"announcementbar|announcement-bar|marquee|leaderboard|adslot|"
            r"ad-slot|adunit|ad-unit|adcontainer|ad-container|adwrapper|"
            r"ad-wrapper|gptad|dfpad|adsbygoogle)",
            re.IGNORECASE,
        )

        # Lines that are pure boilerplate noise even after DOM cleaning
        # (common on tutorial/news sites where these render as plain text
        # rather than being isolated in their own tagged container).
        self.boilerplate_line_pattern = re.compile(
            r"^(share|tweet|pin\s*it|print|advertisement|sponsored|subscribe"
            r"|sign up|sign in|log in|related articles?|read more|next article"
            r"|previous article|trending now|recommended for you|comments?"
            r"|leave a reply|follow us|like us on facebook|jump to content"
            r"|table of contents|edit|talk|\[edit\])$",
            re.IGNORECASE,
        )

        # Wikipedia-specific elements to remove before conversion.
        self.wikipedia_kill_selectors = [
            "table.infobox", "table.navbox", "table.vertical-navbox",
            "table.metadata", "table.ambox", "table.sistersitebox",
            "div.navbox", "div.hatnote", "div.toc", "div#toc",
            "div.catlinks", "div.printfooter", "div.reflist",
            "div.thumb", "div.magnify", "div.noprint",
            "ol.references", "sup.reference", "sup.noprint",
            "span.mw-editsection", "span.mw-cite-backlink",
            "style", "div.reference",
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _clean_url(self, source: str) -> str:
        """Strips tracking and recommendation query parameters from URL."""
        parsed = urllib.parse.urlparse(source.strip())
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

    def _get_domain(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc.lower()

    def _warm_up_session(self, url: str) -> None:
        """
        Best-effort: hits the site's homepage once to pick up any anti-bot
        verification cookies before requesting the actual page. Some WAFs
        reject a client's very first request on a deep URL but allow it once
        a session cookie exists. Failures here are ignored — this is purely
        an attempt to look more like a real browsing session, not a
        requirement for extraction to proceed.
        """
        parsed = urllib.parse.urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}/"
        try:
            self.session.get(root, headers=self.headers, verify=self.verify_ssl, timeout=self.timeout)
        except Exception:
            pass

    def _looks_like_bot_block(self, raw_html: str) -> bool:
        return bool(self.bot_block_patterns.search(raw_html[:5000]))

    def _fix_encoding(self, text: str) -> str:
        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\xa0\u200b\u200e\u200f\ufeff]", " ", text)
        # Collapse leftover citation brackets like "[1]" "[citation needed]"
        text = re.sub(r"\[\s*\d+\s*\]", "", text)
        text = re.sub(r"\[\s*citation needed\s*\]", "", text, flags=re.IGNORECASE)
        return text

    def _format_markdown_output(self, text: str) -> str:
        """
        Collapses excess blank lines and drops boilerplate lines OUTSIDE
        fenced code blocks, while preserving exact formatting (indentation,
        internal blank lines) INSIDE fenced code blocks (```...```).
        Applying line.strip() / blank-line-collapsing indiscriminately (as a
        naive cleanup pass would) silently destroys code snippet formatting,
        which matters a lot for tutorial sites like GFG/W3Schools/JavaTpoint.
        """
        lines = text.splitlines()
        out = []
        in_code_block = False
        blank_pending = False

        for raw_line in lines:
            stripped = raw_line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                out.append(raw_line.rstrip())
                blank_pending = False
                continue

            if in_code_block:
                out.append(raw_line.rstrip("\n"))
                continue

            if not stripped:
                blank_pending = True
                continue

            if self.boilerplate_line_pattern.match(stripped):
                continue

            if blank_pending and out:
                out.append("")
            out.append(stripped)
            blank_pending = False

        return "\n".join(out).strip()

    def _remove_noise_elements(self, soup: BeautifulSoup) -> None:
        """Mutates soup in place, decomposing junk tags and noise-classed elements."""
        for tag in soup(self.junk_tags):
            tag.decompose()

        for element in list(soup.find_all(True)):
            if getattr(element, "attrs", None) is None:
                continue
            # Never strip the article/main container itself, even if a
            # generic keyword happens to appear in its id/class.
            if element.name in ("article", "main"):
                continue
            element_id = str(element.attrs.get("id") or "")
            raw_class = element.attrs.get("class") or []
            element_class = " ".join(raw_class) if isinstance(raw_class, list) else str(raw_class)
            role = str(element.attrs.get("role") or "")

            if (
                self.noise_word_boundary_patterns.search(element_id)
                or self.noise_word_boundary_patterns.search(element_class)
                or self.noise_substring_patterns.search(element_id)
                or self.noise_substring_patterns.search(element_class)
                or role.lower() in ("navigation", "banner", "complementary", "contentinfo")
                or element.attrs.get("aria-hidden") == "true"
            ):
                element.decompose()

    # ------------------------------------------------------------------ #
    # JSON-LD (news / blogs)
    # ------------------------------------------------------------------ #

    def _iter_json_ld_items(self, data):
        """Recursively yields flat dict items, expanding lists and @graph wrappers."""
        if isinstance(data, list):
            for entry in data:
                yield from self._iter_json_ld_items(entry)
        elif isinstance(data, dict):
            if isinstance(data.get("@graph"), list):
                yield from self._iter_json_ld_items(data["@graph"])
            else:
                yield data

    def _extract_json_ld_article_body(self, soup: BeautifulSoup) -> str:
        """Extracts 'articleBody' from NewsArticle/Article/BlogPosting JSON-LD, incl. @graph."""
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
            except Exception:
                continue

            for item in self._iter_json_ld_items(data):
                if not isinstance(item, dict):
                    continue
                item_type = item.get("@type")
                types = item_type if isinstance(item_type, list) else [item_type]
                if not any(t in ["NewsArticle", "Article", "BlogPosting", "TechArticle", "WebPage"] for t in types):
                    continue

                body = item.get("articleBody")
                if isinstance(body, list):
                    body = "\n\n".join(str(p) for p in body)
                if body and isinstance(body, str) and len(body.strip()) > 200:
                    headline = item.get("headline", "")
                    prefix = f"# {headline}\n\n" if headline else ""
                    return prefix + body.strip()
        return ""

    # ------------------------------------------------------------------ #
    # DOM extraction (generic: tutorials, blogs, fallback for news)
    # ------------------------------------------------------------------ #

    def _strip_link_dense_blocks(self, node, link_density_threshold: float = 0.6, min_links: int = 4) -> None:
        """
        Removes block-level descendants that are mostly hyperlinks — the
        structural signature of sidebars, "related/recommended" rails,
        tag clouds, and footer link grids — regardless of what they're
        named in the markup. This catches sidebar/banner content that
        survives class/id keyword filtering because the site uses
        unrecognizable or randomly-hashed class names.

        A block is only removed if it's link-dominated AND has very little
        real paragraph prose, so a genuine content section that happens to
        contain several inline citations/links is left alone.
        """
        for block in list(node.find_all(["div", "section", "ul", "nav", "aside"])):
            if block.parent is None:  # already removed by an earlier decompose in this loop
                continue
            links = block.find_all("a")
            if len(links) < min_links:
                continue
            total_text_len = len(block.get_text(strip=True))
            if total_text_len == 0:
                continue
            link_text_len = sum(len(a.get_text(strip=True)) for a in links)
            density = link_text_len / total_text_len
            paragraph_text_len = sum(len(p.get_text(strip=True)) for p in block.find_all("p"))
            if density >= link_density_threshold and paragraph_text_len < 150:
                block.decompose()

    def _find_best_container(self, h1_tag, min_text_threshold: int = 150):
        """
        Climbs upward from the heading's parent and returns the FIRST
        ancestor whose subtree contains at least `min_text_threshold` chars
        of substantial paragraph/list-item text.

        Note this deliberately stops at the first ancestor that clears the
        threshold rather than the one with the highest score: since a parent
        always contains everything its children do, "highest score" is
        monotonically non-decreasing while climbing and would almost always
        resolve all the way to <body> — defeating the purpose. Stopping
        early keeps us at the smallest container that plausibly holds the
        full article, which is what actually excludes unrelated sibling
        sections (nav megamenus, footer widgets) sitting elsewhere in <body>.
        """
        node = h1_tag.parent
        last_node = node
        depth = 0
        while node is not None and getattr(node, "name", None) not in (None, "[document]") and depth < 10:
            text_blocks = node.find_all(["p", "li", "td"])
            score = sum(len(t.get_text(strip=True)) for t in text_blocks if len(t.get_text(strip=True)) > 40)
            last_node = node
            if score >= min_text_threshold:
                return node
            if node.name == "body":
                break
            node = node.parent
            depth += 1
        return last_node

    def _extract_from_dom(self, soup: BeautifulSoup) -> str:
        """Parses and converts core HTML article DOM to clean Markdown."""
        dom_copy = BeautifulSoup(str(soup), "html.parser")

        self._remove_noise_elements(dom_copy)

        article_node = (
            dom_copy.find("article")
            or dom_copy.find(class_=re.compile(r"article|article-body|story|entry-content|main-content|post-content", re.I))
            or dom_copy.find("main")
        )

        if not article_node:
            h1_tag = dom_copy.find("h1")
            if h1_tag:
                article_node = self._find_best_container(h1_tag)

        if article_node:
            self._strip_link_dense_blocks(article_node)

        # Safety net: whatever we picked (including the h1/article/main
        # branches above), if it still contains very little real text, fall
        # back to <body> rather than returning near-nothing — a noisy full
        # page beats a near-empty extraction the caller can't use at all.
        body_node = dom_copy.find("body") or dom_copy
        current_len = len(article_node.get_text(strip=True)) if article_node else 0
        if current_len < 200 and len(body_node.get_text(strip=True)) > current_len:
            self._strip_link_dense_blocks(body_node)
            article_node = body_node

        markdown_text = self.converter.convert(str(article_node))
        return self._format_markdown_output(markdown_text)

    # ------------------------------------------------------------------ #
    # Wikipedia-specific extraction
    # ------------------------------------------------------------------ #

    def _extract_wikipedia(self, soup: BeautifulSoup) -> str:
        content = soup.find("div", id="mw-content-text") or soup.find("div", class_="mw-parser-output")
        if not content:
            return ""

        dom_copy = BeautifulSoup(str(content), "html.parser")

        for selector in self.wikipedia_kill_selectors:
            for tag in dom_copy.select(selector):
                tag.decompose()

        for tag in dom_copy(self.junk_tags):
            tag.decompose()

        # Stop at "See also" / "References" / "External links" sections —
        # everything after is boilerplate for RAG purposes.
        stop_headings = re.compile(r"^(see also|references|external links|notes|further reading|bibliography)$", re.I)
        for heading in dom_copy.find_all(re.compile(r"^h[1-6]$")):
            if stop_headings.match(heading.get_text(strip=True)):
                for sibling in list(heading.find_next_siblings()):
                    sibling.decompose()
                heading.decompose()

        self._strip_link_dense_blocks(dom_copy)

        markdown_text = self.converter.convert(str(dom_copy))
        return self._format_markdown_output(markdown_text)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def extract(self, source: str) -> str:
        url = self._clean_url(source)
        domain = self._get_domain(url)

        try:
            if domain not in self._warmed_domains:
                self._warm_up_session(url)
                self._warmed_domains.add(domain)

            response = self.session.get(
                url,
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                raise ValueError(f"Unexpected content-type '{content_type}' (not HTML).")

            response.encoding = response.apparent_encoding or response.encoding
            soup = BeautifulSoup(response.text, "html.parser")

            # --- SMART SELECTION ROUTER ---
            if "wikipedia.org" in domain:
                final_text = self._fix_encoding(self._extract_wikipedia(soup))
            else:
                dom_text = self._fix_encoding(self._extract_from_dom(soup))
                json_ld_body = self._fix_encoding(self._extract_json_ld_article_body(soup))

                # Check if DOM content has rich tutorial elements (code blocks or markdown tables)
                has_code_or_tables = bool(re.search(r"```|\|[\s\-:]+\|", dom_text))

                if json_ld_body and not has_code_or_tables:
                    # News / standard article where JSON-LD provides the full, un-paywalled text
                    final_text = json_ld_body
                else:
                    # Tutorials (GFG/W3Schools/JavaTpoint) or sites without valid JSON-LD articleBody
                    final_text = dom_text if dom_text.strip() else json_ld_body

            if len(final_text.strip()) < 100:
                raw_len = len(response.text)
                if self._looks_like_bot_block(response.text):
                    raise ValueError(
                        f"Site returned an anti-bot / JS-challenge page (raw response {raw_len} chars), "
                        "not the real article. This is a WAF (Cloudflare/Akamai/PerimeterX-style) blocking "
                        "plain 'requests' traffic — headers/retries can't reliably bypass this. Use a "
                        "headless browser (Playwright/Selenium) for this domain, or a TLS-fingerprint-aware "
                        "client (e.g. curl_cffi, cloudscraper)."
                    )
                elif raw_len > 20000:
                    raise ValueError(
                        f"Received a full page ({raw_len} raw chars) but extraction produced almost no "
                        "usable text. This points to a selector/noise-removal mismatch for this site's "
                        "markup (not a network block) — inspect the page's DOM structure directly."
                    )
                else:
                    raise ValueError(
                        f"Response was unusually small ({raw_len} raw chars) — likely blocked, "
                        "redirected, geo-restricted, or rate-limited by the server."
                    )

            return final_text

        except Exception as e:
            raise RuntimeError(f"WebExtractor failed for '{url}': {str(e)}")