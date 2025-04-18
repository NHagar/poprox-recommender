from itertools import zip_longest

from lenskit.pipeline import Component

from poprox_concepts.domain import RecommendationList


class Concatenate(Component):
    config: None

    def __call__(self, recs1: RecommendationList, recs2: RecommendationList) -> RecommendationList:
        """
        Concatenates two sets of candidates, while deduplicating them, keeping the
        first occurrence of each article (by id), and maintaining their original order.

        This is achieved by inserting articles into a dict in reverse order, so that
        articles from the second candidate set are written first and then overwritten
        by articles from the first candidate set (if there are collisions.) Afterward,
        the dict keys can be ignored and the dict values are the deduplicated candidates
        in reverse order. Reversing them one more time returns them to the original order.
        """
        reverse_articles = reversed(recs1.articles + recs2.articles)
        reverse_extras = reversed(recs1.extras + recs2.extras)

        articles = {article.article_id: article for article in reverse_articles}
        extras = {article.article_id: extra for article, extra in zip_longest(reverse_articles, reverse_extras)}

        return RecommendationList(articles=list(reversed(articles.values())), extras=list(reversed(extras.values())))
