"""
Corpus Analysis Module

This module provides tools for analyzing and visualizing corpora.

Features:
- Word frequency analysis
- Term co-occurrence
- Publication trends over time
- Word cloud generation
- Topic distribution

Example:
    >>> from ontorag.corpus import Corpus
    >>> from ontorag.corpus.analysis import CorpusAnalyzer
    >>> 
    >>> corpus = Corpus.load("my_corpus.json")
    >>> analyzer = CorpusAnalyzer(corpus)
    >>> 
    >>> # Get term frequencies
    >>> top_terms = analyzer.term_frequency(n=20)
    >>> 
    >>> # Analyze taxonomy coverage
    >>> coverage = analyzer.taxonomy_coverage(CLIMATE_TAXONOMY)
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .corpus import Corpus

logger = logging.getLogger(__name__)


# Default stopwords for filtering
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'their',
    'they', 'which', 'what', 'who', 'whom', 'where', 'when', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very',
    'also', 'however', 'therefore', 'thus', 'hence', 'study', 'paper',
    'results', 'using', 'based', 'show', 'shows', 'shown', 'use', 'used',
    'new', 'two', 'one', 'first', 'second', 'third', 'between', 'through',
}


class CorpusAnalyzer:
    """
    Analyze and visualize corpus content.
    
    Provides statistical analysis and visualization of document collections.
    
    Example:
        >>> analyzer = CorpusAnalyzer(corpus)
        >>> 
        >>> # Term frequency
        >>> terms = analyzer.term_frequency(n=20)
        >>> for term, count in terms:
        ...     print(f"{term}: {count}")
        >>> 
        >>> # Year distribution
        >>> years = analyzer.year_distribution()
        >>> 
        >>> # Generate word cloud
        >>> analyzer.generate_wordcloud("wordcloud.png")
    """
    
    def __init__(self, corpus: "Corpus"):
        """
        Initialize analyzer with a corpus.
        
        Args:
            corpus: Corpus to analyze
        """
        self.corpus = corpus
    
    def term_frequency(
        self,
        n: int = 50,
        source: str = "both",
        min_length: int = 3,
        stopwords: Set[str] = None
    ) -> List[Tuple[str, int]]:
        """
        Get most frequent terms in corpus.
        
        Args:
            n: Number of top terms to return
            source: "title", "abstract", or "both"
            min_length: Minimum term length
            stopwords: Set of words to exclude
            
        Returns:
            List of (term, count) tuples
        """
        stopwords = stopwords or STOPWORDS
        counter = Counter()
        
        for doc in self.corpus:
            text = ""
            if source in ["title", "both"]:
                text += doc.title + " "
            if source in ["abstract", "both"]:
                text += doc.abstract + " "
            
            # Tokenize and clean
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            words = [w for w in words if len(w) >= min_length and w not in stopwords]
            counter.update(words)
        
        return counter.most_common(n)
    
    def ngram_frequency(
        self,
        n: int = 2,
        top_k: int = 30,
        source: str = "both"
    ) -> List[Tuple[str, int]]:
        """
        Get most frequent n-grams.
        
        Args:
            n: N-gram size (2 for bigrams, 3 for trigrams)
            top_k: Number of top n-grams to return
            source: "title", "abstract", or "both"
            
        Returns:
            List of (ngram, count) tuples
        """
        counter = Counter()
        
        for doc in self.corpus:
            text = ""
            if source in ["title", "both"]:
                text += doc.title + " "
            if source in ["abstract", "both"]:
                text += doc.abstract + " "
            
            # Tokenize
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            words = [w for w in words if w not in STOPWORDS and len(w) > 2]
            
            # Generate n-grams
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                counter[ngram] += 1
        
        return counter.most_common(top_k)
    
    def year_distribution(self) -> Dict[int, int]:
        """
        Get distribution of documents by publication year.
        
        Returns:
            Dictionary mapping year to document count
        """
        counter = Counter()
        
        for doc in self.corpus:
            if doc.year:
                counter[doc.year] += 1
        
        return dict(sorted(counter.items()))
    
    def source_distribution(self) -> Dict[str, int]:
        """
        Get distribution of documents by source.
        
        Returns:
            Dictionary mapping source name to document count
        """
        counter = Counter(doc.source for doc in self.corpus)
        return dict(counter)
    
    def category_distribution(self, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Get most common categories/keywords.
        
        Args:
            top_k: Number of top categories to return
            
        Returns:
            List of (category, count) tuples
        """
        counter = Counter()
        
        for doc in self.corpus:
            counter.update(doc.categories)
            counter.update(doc.keywords)
        
        return counter.most_common(top_k)
    
    def taxonomy_coverage(
        self,
        taxonomy: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Analyze corpus coverage of a taxonomy.
        
        Args:
            taxonomy: Dictionary mapping category names to term lists
            
        Returns:
            Dictionary with:
            - category_counts: Counts per category
            - term_counts: Counts per term
            - coverage_pct: Percentage of documents with matches
            
        Example:
            >>> taxonomy = {
            ...     "hazards": ["flood", "drought", "heat wave"],
            ...     "measures": ["green roof", "early warning"]
            ... }
            >>> coverage = analyzer.taxonomy_coverage(taxonomy)
        """
        # Combine all text
        combined_text = ""
        for doc in self.corpus:
            combined_text += f" {doc.title} {doc.abstract} ".lower()
        
        category_counts = {}
        term_counts = {}
        docs_with_match = 0
        
        for category, terms in taxonomy.items():
            cat_total = 0
            for term in terms:
                count = combined_text.count(term.lower())
                if count > 0:
                    term_counts[term] = count
                    cat_total += count
            category_counts[category] = cat_total
        
        # Count documents with at least one match
        all_terms = [t.lower() for terms in taxonomy.values() for t in terms]
        for doc in self.corpus:
            doc_text = f"{doc.title} {doc.abstract}".lower()
            if any(term in doc_text for term in all_terms):
                docs_with_match += 1
        
        return {
            "category_counts": category_counts,
            "term_counts": term_counts,
            "documents_with_match": docs_with_match,
            "coverage_pct": docs_with_match / len(self.corpus) * 100 if self.corpus else 0,
            "top_terms": sorted(term_counts.items(), key=lambda x: -x[1])[:20],
        }
    
    def citation_analysis(self) -> Dict[str, Any]:
        """
        Analyze citation metrics.
        
        Returns:
            Dictionary with citation statistics
        """
        citations = [doc.citation_count for doc in self.corpus]
        
        if not citations:
            return {"error": "No citation data"}
        
        citations_sorted = sorted(citations, reverse=True)
        
        return {
            "total_citations": sum(citations),
            "mean_citations": sum(citations) / len(citations),
            "median_citations": citations_sorted[len(citations) // 2],
            "max_citations": max(citations),
            "papers_with_citations": sum(1 for c in citations if c > 0),
            "highly_cited_10plus": sum(1 for c in citations if c >= 10),
            "highly_cited_100plus": sum(1 for c in citations if c >= 100),
        }
    
    def generate_wordcloud(
        self,
        output_path: str = "wordcloud.png",
        source: str = "both",
        max_words: int = 100,
        **kwargs
    ) -> str:
        """
        Generate word cloud visualization.
        
        Args:
            output_path: Path to save image
            source: "title", "abstract", or "both"
            max_words: Maximum words in cloud
            **kwargs: Additional WordCloud parameters
            
        Returns:
            Path to saved image
            
        Raises:
            ImportError: If wordcloud or matplotlib not installed
        """
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "wordcloud and matplotlib required. "
                "Install with: pip install wordcloud matplotlib"
            )
        
        # Combine text
        text = ""
        for doc in self.corpus:
            if source in ["title", "both"]:
                text += doc.title + " "
            if source in ["abstract", "both"]:
                text += doc.abstract + " "
        
        # Generate cloud
        wc = WordCloud(
            width=1200,
            height=600,
            max_words=max_words,
            background_color="white",
            stopwords=STOPWORDS,
            colormap="viridis",
            **kwargs
        ).generate(text)
        
        # Save
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word cloud saved to {output_path}")
        return output_path
    
    def plot_year_distribution(
        self,
        output_path: str = "year_distribution.png"
    ) -> str:
        """
        Plot publication year distribution.
        
        Args:
            output_path: Path to save image
            
        Returns:
            Path to saved image
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "matplotlib and seaborn required. "
                "Install with: pip install matplotlib seaborn"
            )
        
        years = self.year_distribution()
        
        plt.figure(figsize=(12, 5))
        sns.set_style("whitegrid")
        
        plt.bar(years.keys(), years.values(), color='steelblue', alpha=0.8)
        plt.xlabel('Year')
        plt.ylabel('Number of Documents')
        plt.title(f'Publication Distribution - {self.corpus.name}')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Year distribution plot saved to {output_path}")
        return output_path
    
    def summary_report(self) -> str:
        """
        Generate a comprehensive text report.
        
        Returns:
            Formatted report string
        """
        stats = self.corpus.stats()
        year_dist = self.year_distribution()
        top_terms = self.term_frequency(n=15)
        citations = self.citation_analysis()
        
        lines = [
            "=" * 70,
            f"CORPUS ANALYSIS REPORT: {self.corpus.name}",
            "=" * 70,
            "",
            "OVERVIEW",
            "-" * 30,
            f"Total Documents: {stats['total_documents']}",
            f"With Abstract: {stats.get('with_abstract', 0)}",
            f"With PDF: {stats.get('with_pdf', 0)}",
            "",
            "SOURCES",
            "-" * 30,
        ]
        
        for source, count in stats.get("by_source", {}).items():
            pct = count / stats['total_documents'] * 100
            lines.append(f"  {source}: {count} ({pct:.1f}%)")
        
        lines.extend([
            "",
            "YEAR RANGE",
            "-" * 30,
        ])
        
        if year_dist:
            min_year = min(year_dist.keys())
            max_year = max(year_dist.keys())
            lines.append(f"  {min_year} - {max_year}")
            
            # Last 5 years
            recent_years = sorted(year_dist.keys())[-5:]
            for year in recent_years:
                lines.append(f"    {year}: {year_dist[year]} papers")
        
        lines.extend([
            "",
            "TOP 15 TERMS",
            "-" * 30,
        ])
        
        for term, count in top_terms:
            lines.append(f"  {term}: {count}")
        
        lines.extend([
            "",
            "CITATIONS",
            "-" * 30,
            f"  Mean: {citations.get('mean_citations', 0):.1f}",
            f"  Median: {citations.get('median_citations', 0)}",
            f"  Max: {citations.get('max_citations', 0)}",
            f"  10+ citations: {citations.get('highly_cited_10plus', 0)}",
            f"  100+ citations: {citations.get('highly_cited_100plus', 0)}",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


__all__ = ["CorpusAnalyzer", "STOPWORDS"]
