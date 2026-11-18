"""Streamlit interface for comic retrieval and generation."""

import json
import re
from pathlib import Path

import streamlit as st

from RAG import ComicRAG


ROOT_DIR = Path(__file__).resolve().parent
COMICS_DIR = ROOT_DIR / "comics"


@st.cache_resource
def get_comic_rag(comics_dir: str) -> ComicRAG:
    return ComicRAG(Path(comics_dir))


st.set_page_config(page_title="Comic finder", layout="wide")
st.title("Find a comic")
st.write("Describe a story to find similar comics, or create a story script for Colab.")

try:
    rag = get_comic_rag(str(COMICS_DIR))
except ImportError as error:
    st.error(str(error))
    st.stop()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

if st.button("Refresh comic index"):
    try:
        with st.spinner("Rebuilding the FAISS index..."):
            indexed_count = rag.reindex()
        if "comic_query" in st.session_state:
            st.session_state["comic_results"] = rag.search(
                st.session_state["comic_query"],
                top_k=st.session_state.get("comic_top_k", 3),
            )
        st.success(f"Indexed {indexed_count} story files.")
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not refresh the comic index: {error}")

with st.form("comic_search"):
    query = st.text_area(
        "What kind of comic are you looking for?",
        placeholder="For example: A child learns to enjoy Diwali safely with a parent.",
    )
    top_k = st.slider("Number of similar comics", min_value=1, max_value=5, value=3)
    submitted = st.form_submit_button("Find similar comics")

if submitted:
    try:
        results = rag.search(query, top_k=top_k)
    except ValueError as error:
        st.session_state.pop("comic_query", None)
        st.session_state.pop("comic_top_k", None)
        st.session_state.pop("comic_results", None)
        st.error(str(error))
    else:
        st.session_state["comic_query"] = query.strip()
        st.session_state["comic_top_k"] = top_k
        st.session_state["comic_results"] = results

if "comic_results" in st.session_state:
    results = st.session_state["comic_results"]
    if results:
        st.subheader("Closest comics")
        for result in results:
            with st.container():
                st.markdown(
                    f"**{result.document.title}** "
                    f"(similarity: {result.score:.3f})"
                )
                if result.document.image_path is not None:
                    st.image(
                        str(result.document.image_path),
                        use_container_width=True,
                    )
                else:
                    st.info("Story indexed; comic image has not been rendered yet.")
                st.caption(result.document.text[:700])
    else:
        st.info("No matching comics were found.")

    st.subheader("Would you like a new story script to render in Colab?")
    generate_choice = st.radio(
        "Create a new story script?",
        ("No", "Yes"),
        horizontal=True,
        key="generate_comic_choice",
    )
    if generate_choice == "Yes":
        filename_prefix = st.text_input(
            "Filename prefix",
            placeholder="e.g. diwali_story",
            help="The generated files will be saved as <prefix>.json and <prefix>.txt.",
        )
        if st.button("Generate story script", type="primary"):
            filename_prefix = filename_prefix.strip()
            reserved_names = {"CON", "PRN", "AUX", "NUL"}
            reserved_names.update(f"COM{number}" for number in range(1, 10))
            reserved_names.update(f"LPT{number}" for number in range(1, 10))
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,79}", filename_prefix):
                st.error(
                    "Use a prefix of 1-80 letters, numbers, spaces, underscores, "
                    "or hyphens. Start with a letter or number."
                )
            elif filename_prefix.upper() in reserved_names:
                st.error("That prefix is reserved by Windows. Choose another.")
            else:
                story_path = COMICS_DIR / f"{filename_prefix}.json"
                plot_path = COMICS_DIR / f"{filename_prefix}.txt"
                if story_path.exists() or plot_path.exists():
                    st.error(
                        f"Files for prefix {filename_prefix!r} already exist. "
                        "Choose another prefix to avoid overwriting them."
                    )
                else:
                    try:
                        from generate_scene import (
                            StoryRequestRefusedError,
                            generate_story,
                        )

                        with st.spinner("Generating the story script..."):
                            story = generate_story(st.session_state["comic_query"])
                            story_path.write_text(
                                json.dumps(story, ensure_ascii=False, indent=4) + "\n",
                                encoding="utf-8",
                            )
                            plot_path.write_text(
                                st.session_state["comic_query"] + "\n",
                                encoding="utf-8",
                            )

                        st.session_state["generated_story_json"] = str(story_path)
                        st.session_state["generated_plot_text"] = str(plot_path)
                        indexed_count = rag.reindex()
                        st.session_state["comic_results"] = rag.search(
                            st.session_state["comic_query"],
                            top_k=st.session_state.get("comic_top_k", 3),
                        )
                        st.success(
                            f"Saved {story_path.name} and {plot_path.name}; "
                            f"reindexed {indexed_count} stories."
                        )
                        st.info(
                            "Render the JSON in Colab. To make the finished comic searchable, "
                            "place its PNG beside the JSON with the same filename stem, then "
                            "click Refresh comic index."
                        )
                    except StoryRequestRefusedError as error:
                        st.warning(str(error))
                    except (OSError, RuntimeError, ValueError) as error:
                        st.error(f"Could not generate, save, or reindex the story: {error}")

if "generated_story_json" in st.session_state:
    story_path = Path(st.session_state["generated_story_json"])
    plot_path = Path(st.session_state["generated_plot_text"])
    if story_path.is_file() and plot_path.is_file():
        st.subheader("Latest story files")
        story_column, plot_column = st.columns(2)
        story_column.download_button(
            "Download story JSON",
            data=story_path.read_bytes(),
            file_name=story_path.name,
            mime="application/json",
            key="download_story_json",
        )
        plot_column.download_button(
            "Download plot text",
            data=plot_path.read_bytes(),
            file_name=plot_path.name,
            mime="text/plain",
            key="download_plot_text",
        )
