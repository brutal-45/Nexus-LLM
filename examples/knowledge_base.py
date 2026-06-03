#!/usr/bin/env python3
"""Knowledge base example."""
from nexus_llm.knowledge import KnowledgeBase, KnowledgeEntry

kb = KnowledgeBase()
kb.add_entry(KnowledgeEntry(id='1', title='Python', content='Python is a programming language', tags=['programming']))
kb.add_entry(KnowledgeEntry(id='2', title='ML', content='Machine learning is a subset of AI', tags=['ai']))

results = kb.search('programming')
for r in results:
    print(f'{r.title}: {r.content}')
