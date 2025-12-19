#!/usr/bin/env python3
"""
OntoRAG Example: Corpus Collection from Academic Sources

This example shows how to:
1. Search multiple academic databases (arXiv, Semantic Scholar, OpenAlex)
2. Build a corpus with automatic deduplication
3. Analyze corpus statistics
4. Export to JSON/CSV

Prerequisites:
    $ pip install ontorag[corpus]

Run this example:
    $ python 03_corpus_collection.py
"""

from ontorag.corpus import (
    Corpus,
    CorpusAnalyzer,
    ArxivSource,
    SemanticScholarSource,
    OpenAlexSource,
)


def main():
    print("=" * 60)
    print("OntoRAG: Corpus Collection")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Create Corpus
    # =========================================================================
    print("\n📚 Step 1: Creating corpus...")
    
    corpus = Corpus(
        name="climate_adaptation_research",
        description="Academic papers on climate adaptation and mitigation"
    )
    
    # =========================================================================
    # STEP 2: Search Academic Sources
    # =========================================================================
    print("\n🔍 Step 2: Searching academic sources...")
    
    # Define search query
    query = "climate adaptation flood mitigation"
    
    # Search arXiv
    print(f"\n   Searching arXiv for: '{query}'...")
    try:
        arxiv = ArxivSource()
        arxiv_docs = arxiv.search(
            query=query,
            max_results=20,
            sort_by="relevance"
        )
        print(f"   Found {len(arxiv_docs)} papers on arXiv")
        
        # Add to corpus
        for doc in arxiv_docs:
            corpus.add(doc)
    except ImportError as e:
        print(f"   arXiv not available: {e}")
    except Exception as e:
        print(f"   arXiv search failed: {e}")
    
    # Search Semantic Scholar
    print(f"\n   Searching Semantic Scholar...")
    try:
        ss = SemanticScholarSource()
        ss_docs = ss.search(
            query=query,
            max_results=20,
            year_range=(2020, 2024)
        )
        print(f"   Found {len(ss_docs)} papers on Semantic Scholar")
        
        for doc in ss_docs:
            corpus.add(doc)
    except ImportError as e:
        print(f"   Semantic Scholar not available: {e}")
    except Exception as e:
        print(f"   Semantic Scholar search failed: {e}")
    
    # Search OpenAlex
    print(f"\n   Searching OpenAlex...")
    try:
        openalex = OpenAlexSource(email="researcher@example.com")
        oa_docs = openalex.search(
            query=query,
            max_results=20,
            from_year=2020,
            open_access_only=True
        )
        print(f"   Found {len(oa_docs)} papers on OpenAlex")
        
        for doc in oa_docs:
            corpus.add(doc)
    except ImportError as e:
        print(f"   OpenAlex not available: {e}")
    except Exception as e:
        print(f"   OpenAlex search failed: {e}")
    
    # =========================================================================
    # STEP 3: Corpus Statistics
    # =========================================================================
    print("\n📊 Step 3: Corpus statistics...")
    
    stats = corpus.stats()
    print(f"\n   Total documents: {stats['total_documents']}")
    print(f"   With abstract: {stats.get('with_abstract', 0)}")
    print(f"   With PDF: {stats.get('with_pdf', 0)}")
    
    if stats.get('by_source'):
        print("\n   By source:")
        for source, count in stats['by_source'].items():
            print(f"   - {source}: {count}")
    
    if stats.get('year_range'):
        print(f"\n   Year range: {stats['year_range'][0]} - {stats['year_range'][1]}")
    
    print(f"   Average citations: {stats.get('avg_citations', 0):.1f}")
    
    # =========================================================================
    # STEP 4: Analyze Corpus
    # =========================================================================
    print("\n🔬 Step 4: Analyzing corpus...")
    
    if len(corpus) > 0:
        analyzer = CorpusAnalyzer(corpus)
        
        # Top terms
        print("\n   Top 10 terms:")
        top_terms = analyzer.term_frequency(n=10)
        for term, count in top_terms:
            print(f"   - {term}: {count}")
        
        # Year distribution
        print("\n   Year distribution:")
        years = analyzer.year_distribution()
        for year, count in sorted(years.items())[-5:]:
            print(f"   - {year}: {count}")
        
        # Category distribution
        print("\n   Top categories:")
        categories = analyzer.category_distribution(top_k=5)
        for cat, count in categories:
            print(f"   - {cat}: {count}")
    
    # =========================================================================
    # STEP 5: Filter Documents
    # =========================================================================
    print("\n🔎 Step 5: Filtering documents...")
    
    # Filter recent papers with high citations
    recent_cited = corpus.filter(
        year_min=2022,
        min_citations=5,
        has_abstract=True
    )
    print(f"   Recent highly-cited papers (2022+, 5+ citations): {len(recent_cited)}")
    
    # Show top 3
    if recent_cited:
        print("\n   Top 3 by citations:")
        sorted_docs = sorted(recent_cited, key=lambda d: d.citation_count, reverse=True)
        for doc in sorted_docs[:3]:
            title = doc.title[:60] + "..." if len(doc.title) > 60 else doc.title
            print(f"   - [{doc.citation_count}] {title}")
    
    # =========================================================================
    # STEP 6: Export Corpus
    # =========================================================================
    print("\n💾 Step 6: Exporting corpus...")
    
    if len(corpus) > 0:
        # Save as JSON
        corpus.save("climate_corpus.json")
        print("   Saved to: climate_corpus.json")
        
        # Save as CSV
        corpus.to_csv("climate_corpus.csv")
        print("   Saved to: climate_corpus.csv")
    else:
        print("   Corpus is empty, nothing to export")
    
    # =========================================================================
    # STEP 7: Sample Documents
    # =========================================================================
    print("\n📄 Step 7: Sample documents...")
    
    if len(corpus) > 0:
        print(f"\n   First 3 documents:")
        for i, doc in enumerate(list(corpus)[:3], 1):
            print(f"\n   {i}. {doc.title[:70]}...")
            print(f"      Source: {doc.source}")
            print(f"      Year: {doc.year}")
            print(f"      Citations: {doc.citation_count}")
            if doc.abstract:
                print(f"      Abstract: {doc.abstract[:150]}...")
    
    # =========================================================================
    # Alternative: Use Corpus.search_all()
    # =========================================================================
    print("\n" + "-" * 60)
    print("💡 Alternative: Using Corpus.search_all()")
    print("-" * 60)
    
    print("""
    # Simpler approach - search all sources at once:
    
    corpus = Corpus(name="my_research")
    corpus.search_all(
        query="urban heat island mitigation",
        sources=["arxiv", "semantic_scholar", "openalex"],
        max_per_source=50
    )
    
    print(corpus.summary())
    corpus.save("my_corpus.json")
    """)
    
    print("\n" + "=" * 60)
    print("✅ Corpus collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
