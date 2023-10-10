from typing import Any

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


class ToggleButton:

    def __init__(self, dg: DeltaGenerator, label: str, key: str, **kwargs: Any) -> None:
        self._dg = dg
        self._key = key

        def toggle():
            st.session_state[f"{self._key}-state"] = not self.state

        text = "Hide " if self.state else "Show "
        text += label
        self._dg.button(text,
                        key=f"{self._key}",
                        type="primary" if self.state else "secondary",
                        on_click=toggle,
                        **kwargs)

    @property
    def state(self) -> bool:
        return st.session_state.get(f"{self._key}-state", False)
