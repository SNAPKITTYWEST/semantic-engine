# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phases 15–16: Artifact engine, Mustache + Go-template backends."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ir import Artifact, Provenance, SemanticIR, TemporalState
from .types import ArtifactType, FSMState, PipelineError, VerificationState


# ── Mustache renderer (no external dep required) ─────────────────────────────

class MustacheRenderer:
    """
    Minimal Mustache renderer supporting:
      {{variable}}
      {{#section}} ... {{/section}}   (truthy sections and list iteration)
      {{^section}} ... {{/section}}   (inverted)
    """

    def render(self, template: str, ctx: dict[str, Any]) -> str:
        result = self._render_sections(template, ctx)
        result = self._render_variables(result, ctx)
        return result

    def _render_variables(self, tmpl: str, ctx: dict) -> str:
        def replace(m):
            key = m.group(1).strip()
            val = ctx.get(key, "")
            return str(val) if val is not None else ""
        return re.sub(r"\{\{([^#^/][^}]*)\}\}", replace, tmpl)

    def _render_sections(self, tmpl: str, ctx: dict) -> str:
        # Positive sections: {{#key}} ... {{/key}}
        pattern = re.compile(r"\{\{#(\w+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)
        def replace_section(m):
            key = m.group(1)
            body = m.group(2)
            value = ctx.get(key)
            if not value:
                return ""
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        parts.append(self.render(body, {**ctx, **item}))
                    else:
                        parts.append(self.render(body, {**ctx, ".": item}))
                return "".join(parts)
            return self.render(body, ctx)

        while pattern.search(tmpl):
            tmpl = pattern.sub(replace_section, tmpl)

        # Inverted sections: {{^key}} ... {{/key}}
        inv_pattern = re.compile(r"\{\{\^(\w+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)
        def replace_inverted(m):
            key = m.group(1)
            body = m.group(2)
            value = ctx.get(key)
            return "" if value else self.render(body, ctx)

        while inv_pattern.search(tmpl):
            tmpl = inv_pattern.sub(replace_inverted, tmpl)

        return tmpl

    def render_file(self, template_path: Path, ctx: dict) -> str:
        if not template_path.exists():
            raise PipelineError(FSMState.ARTIFACT_SELECTION, f"Template not found: {template_path}")
        return self.render(template_path.read_text(), ctx)


# ── Go template renderer (subset) ────────────────────────────────────────────

class GoTemplateRenderer:
    """
    Minimal Go template subset renderer:
      {{.Field}}
      {{range .Items}} ... {{end}}
      {{if .Cond}} ... {{end}}
    """

    def render(self, template: str, data: Any) -> str:
        result = self._render_range(template, data)
        result = self._render_if(result, data)
        result = self._render_dot_vars(result, data)
        return result

    def _get(self, data: Any, key: str) -> Any:
        if isinstance(data, dict):
            return data.get(key, "")
        return getattr(data, key, "")

    def _render_dot_vars(self, tmpl: str, data: Any) -> str:
        def replace(m):
            key = m.group(1).strip()
            if key.startswith("."):
                key = key[1:]
            val = self._get(data, key) if key else data
            return str(val) if val is not None else ""
        return re.sub(r"\{\{(\.?\w+)\}\}", replace, tmpl)

    def _render_range(self, tmpl: str, data: Any) -> str:
        pattern = re.compile(r"\{\{range \.(\w+)\}\}(.*?)\{\{end\}\}", re.DOTALL)
        def replace(m):
            key = m.group(1)
            body = m.group(2)
            items = self._get(data, key)
            if not items:
                return ""
            if not isinstance(items, (list, tuple)):
                items = [items]
            parts = []
            for item in items:
                parts.append(self.render(body, item))
            return "".join(parts)
        while pattern.search(tmpl):
            tmpl = pattern.sub(replace, tmpl)
        return tmpl

    def _render_if(self, tmpl: str, data: Any) -> str:
        pattern = re.compile(r"\{\{if \.(\w+)\}\}(.*?)\{\{end\}\}", re.DOTALL)
        def replace(m):
            key = m.group(1)
            body = m.group(2)
            value = self._get(data, key)
            return self.render(body, data) if value else ""
        while pattern.search(tmpl):
            tmpl = pattern.sub(replace, tmpl)
        return tmpl

    def render_file(self, template_path: Path, data: Any) -> str:
        if not template_path.exists():
            raise PipelineError(FSMState.ARTIFACT_SELECTION, f"Template not found: {template_path}")
        return self.render(template_path.read_text(), data)


# ── Artifact engine ───────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


class ArtifactEngine:
    def __init__(self):
        self._mustache = MustacheRenderer()
        self._go       = GoTemplateRenderer()

    def build_prompt(self, ir: SemanticIR, extra: dict | None = None) -> Artifact:
        ctx = self._ir_to_mustache_ctx(ir)
        if extra:
            ctx.update(extra)
        template_path = TEMPLATES_DIR / "prompt.mustache"
        template_text = template_path.read_text() if template_path.exists() else "{{title}}\n\n{{semantic_description}}"
        content = self._mustache.render(template_text, ctx)
        return Artifact.create(
            artifact_type=ArtifactType.PROMPT,
            semantic_root=ir.ir_id,
            template="prompt.mustache",
            content=content,
            provenance=ir.provenance,
            temporal_state=ir.root.temporal_state,
        )

    def build_document(self, ir: SemanticIR, extra: dict | None = None) -> Artifact:
        ctx = self._ir_to_mustache_ctx(ir)
        if extra:
            ctx.update(extra)
        template_path = TEMPLATES_DIR / "document.mustache"
        template_text = template_path.read_text() if template_path.exists() else "# {{title}}\n\n{{semantic_description}}"
        content = self._mustache.render(template_text, ctx)
        return Artifact.create(
            artifact_type=ArtifactType.DOCUMENT,
            semantic_root=ir.ir_id,
            template="document.mustache",
            content=content,
            provenance=ir.provenance,
            temporal_state=ir.root.temporal_state,
        )

    def build_backend(self, ir: SemanticIR, extra: dict | None = None) -> Artifact:
        data = self._ir_to_go_data(ir)
        if extra:
            if isinstance(data, dict):
                data.update(extra)
        template_path = TEMPLATES_DIR / "backend.go.tmpl"
        template_text = template_path.read_text() if template_path.exists() else "// {{.SemanticRoot}}"
        content = self._go.render(template_text, data)
        return Artifact.create(
            artifact_type=ArtifactType.BACKEND,
            semantic_root=ir.ir_id,
            template="backend.go.tmpl",
            content=content,
            provenance=ir.provenance,
            temporal_state=ir.root.temporal_state,
        )

    def _ir_to_mustache_ctx(self, ir: SemanticIR) -> dict:
        return {
            "title":                ir.intent.value.upper(),
            "semantic_description": f"Intent: {ir.intent.value} | Nodes: {len(ir.nodes)}",
            "constraints":          [c.description for n in ir.nodes for c in n.constraints],
            "operations":           [n.operation.name for n in ir.nodes if n.operation],
            "geometry":             [str(n.geometry) for n in ir.nodes if n.geometry and n.geometry.row],
            "provenance":           [ir.provenance.stage],
            "code":                 None,
            "tests":                None,
        }

    def _ir_to_go_data(self, ir: SemanticIR) -> dict:
        return {
            "SemanticRoot": ir.ir_id,
            "Tau":          ir.root.temporal_state.tau,
            "Verified":     ir.verification_state == VerificationState.VERIFIED,
            "Package":      "generated",
            "Types":        [],
            "Operations":   [
                {
                    "Name":       (n.operation.name.capitalize() if n.operation else "NoOp"),
                    "SemanticID": n.id,
                    "Params":     [{"Name": o.name, "Type": o.type_tag} for o in n.operands],
                    "ReturnType": "interface{}",
                    "Constraints": [c.description for c in n.constraints],
                    "Body":       f'\treturn nil, fmt.Errorf("not implemented: {n.id}")',
                }
                for n in ir.nodes if n.operation
            ],
            "Tests": [],
        }
