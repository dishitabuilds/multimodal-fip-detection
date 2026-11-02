"""Explainable, evidence-aware inspection independent of model class labels."""

from .promotion_audit import AUDIT_VERSION, audit_promotion

__all__ = ["AUDIT_VERSION", "audit_promotion"]
