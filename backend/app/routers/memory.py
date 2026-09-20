from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import select

from app.db import get_session
from app.models import Find, Item, ReferenceImage, Search, SearchStatus
from app.routers.searches import SearchCreateResponse
from app.services.search_worker import start_worker

router = APIRouter(prefix="/api")


class FindListItem(BaseModel):
    find_id: str
    item_id: str
    item_name: str
    image_url: str
    crop_url: str
    found_at: datetime


class ItemReferenceImageOut(BaseModel):
    id: str
    url: str


class ItemFindOut(BaseModel):
    find_id: str
    image_url: str
    crop_url: str
    found_at: datetime


class ItemDetailResponse(BaseModel):
    item_id: str
    name: str
    reference_images: list[ItemReferenceImageOut]
    finds: list[ItemFindOut]


@router.get("/finds", response_model=list[FindListItem])
def list_finds() -> list[FindListItem]:
    with get_session() as session:
        rows = session.exec(
            select(Find, Item).join(Item, Find.item_id == Item.id).order_by(Find.found_at.desc())
        ).all()

        return [
            FindListItem(
                find_id=find.id,
                item_id=item.id,
                item_name=item.name,
                image_url=f"/media/{find.full_image_path}",
                crop_url=f"/media/{find.crop_path}",
                found_at=find.found_at,
            )
            for find, item in rows
        ]


@router.get("/items/{item_id}", response_model=ItemDetailResponse)
def get_item(item_id: str) -> ItemDetailResponse:
    with get_session() as session:
        item = session.get(Item, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")

        reference_images = session.exec(
            select(ReferenceImage).where(ReferenceImage.item_id == item_id)
        ).all()

        finds = session.exec(
            select(Find).where(Find.item_id == item_id).order_by(Find.found_at.desc())
        ).all()

        return ItemDetailResponse(
            item_id=item.id,
            name=item.name,
            reference_images=[
                ItemReferenceImageOut(id=r.id, url=f"/media/{r.image_path}") for r in reference_images
            ],
            finds=[
                ItemFindOut(
                    find_id=f.id,
                    image_url=f"/media/{f.full_image_path}",
                    crop_url=f"/media/{f.crop_path}",
                    found_at=f.found_at,
                )
                for f in finds
            ],
        )


@router.post("/items/{item_id}/search", response_model=SearchCreateResponse)
def find_again(item_id: str) -> SearchCreateResponse:
    """Spec 18: reuse an existing Item -- same target text, same reference
    images (ReferenceImage rows are already keyed by item_id, not search_id,
    so a new Search against the same item_id picks them up automatically;
    nothing is re-uploaded or duplicated on disk)."""
    with get_session() as session:
        item = session.get(Item, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found")

        reference_count = len(
            session.exec(select(ReferenceImage).where(ReferenceImage.item_id == item_id)).all()
        )

        search = Search(item_id=item.id, target_text=item.name, status=SearchStatus.SEARCHING)
        session.add(search)
        session.commit()
        session.refresh(search)

        response = SearchCreateResponse(
            search_id=search.id,
            item_id=item.id,
            target_text=search.target_text,
            reference_count=reference_count,
            status=search.status.value,
        )

    start_worker(response.search_id)
    return response
