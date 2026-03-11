"""Create detections table

Revision ID: 001_create_detections
Revises: 
Create Date: 2026-01-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_create_detections"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "detections",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=True),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(100), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("image_path", sa.String(500), nullable=True),
        sa.Column("source_type", sa.Enum("image", "video", "webcam", name="sourcetype"), nullable=False),
    )
    
    # Create indexes
    op.create_index("ix_detections_timestamp", "detections", ["timestamp"])
    op.create_index("ix_detections_camera_id", "detections", ["camera_id"])
    op.create_index("ix_detections_class_name", "detections", ["class_name"])


def downgrade() -> None:
    op.drop_index("ix_detections_class_name")
    op.drop_index("ix_detections_camera_id")
    op.drop_index("ix_detections_timestamp")
    op.drop_table("detections")
