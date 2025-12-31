"""Markdown document loader strategy."""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from src.loaders.base import DocumentLoaderStrategy

logger = logging.getLogger(__name__)


class MarkdownLoaderStrategy(DocumentLoaderStrategy):
    """Strategy for loading Markdown documents."""

    def load(self, file_path: Path) -> List[Document]:
        """Load documents from a Markdown file."""
        logger.info(f"Loading Markdown file: {file_path.name}")
        loader = UnstructuredMarkdownLoader(str(file_path))
        return loader.load()

    def supports(self, file_path: Path) -> bool:
        """Check if file is a Markdown file."""
        return file_path.suffix.lower() == ".md"
