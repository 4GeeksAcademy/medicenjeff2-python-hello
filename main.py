import csv
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field


PRODUCTS_FILE = Path(__file__).resolve().parent / "products.csv"
FIELDNAMES = ["id", "name", "quantity", "unit", "price"]
inventory_lock = Lock()


class InventoryInput(BaseModel):
	name: str = Field(min_length=1, max_length=200, pattern=r"\S")
	quantity: int = Field(ge=0)
	unit: str = Field(min_length=1, max_length=50, pattern=r"\S")


class InventoryProduct(InventoryInput):
	id: int


class StockAdjustment(BaseModel):
	delta: int = Field(strict=True, description="Positivo para entradas, negativo para salidas")


class ProductInput(InventoryInput):
	unit: str = Field(default="unidades", min_length=1, max_length=50, pattern=r"\S")
	price: float = Field(ge=0, allow_inf_nan=False)


class Product(ProductInput):
	id: int


def read_products() -> list[Product]:
	if not PRODUCTS_FILE.exists():
		return []
	with PRODUCTS_FILE.open(newline="", encoding="utf-8") as csv_file:
		return [Product.model_validate(row) for row in csv.DictReader(csv_file)]


def write_products(products: list[Product]) -> None:
	temporary_path = None
	try:
		with tempfile.NamedTemporaryFile(
			mode="w", newline="", encoding="utf-8",
			dir=PRODUCTS_FILE.parent, delete=False,
		) as csv_file:
			temporary_path = Path(csv_file.name)
			writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
			writer.writeheader()
			writer.writerows(product.model_dump() for product in products)
		os.replace(temporary_path, PRODUCTS_FILE)
	finally:
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
	with inventory_lock:
		if not PRODUCTS_FILE.exists():
			write_products([])
	yield


app = FastAPI(title="API de inventario", lifespan=lifespan)


@app.get("/inventory", response_model=list[InventoryProduct])
@app.get("/products", response_model=list[Product])
def list_products():
	with inventory_lock:
		return read_products()


@app.post("/inventory", response_model=InventoryProduct, status_code=201)
def create_inventory_product(data: InventoryInput):
	return create_product(ProductInput(price=0, **data.model_dump()))


@app.get("/inventory/alerts", response_model=list[InventoryProduct])
def inventory_alerts(threshold: int = Query(default=10, ge=0)):
	with inventory_lock:
		return [product for product in read_products() if product.quantity < threshold]


@app.patch("/inventory/{product_id}", response_model=InventoryProduct)
def adjust_stock(product_id: int, data: StockAdjustment):
	with inventory_lock:
		products = read_products()
		for index, product in enumerate(products):
			if product.id == product_id:
				quantity = product.quantity + data.delta
				if quantity < 0:
					raise HTTPException(
						status_code=409,
						detail=f"Stock insuficiente: hay {product.quantity} {product.unit}; "
						f"no se pueden retirar {-data.delta}.",
					)
				updated_product = product.model_copy(update={"quantity": quantity})
				products[index] = updated_product
				write_products(products)
				return updated_product
	raise HTTPException(status_code=404, detail="Producto no encontrado")


@app.get("/products/{product_id}", response_model=Product)
def get_product(product_id: int):
	with inventory_lock:
		for product in read_products():
			if product.id == product_id:
				return product
	raise HTTPException(status_code=404, detail="Producto no encontrado")


@app.post("/products", response_model=Product, status_code=201)
def create_product(data: ProductInput):
	with inventory_lock:
		products = read_products()
		product_id = max((product.id for product in products), default=0) + 1
		product = Product(id=product_id, **data.model_dump())
		products.append(product)
		write_products(products)
		return product


@app.put("/products/{product_id}", response_model=Product)
def update_product(product_id: int, data: ProductInput):
	with inventory_lock:
		products = read_products()
		for index, product in enumerate(products):
			if product.id == product_id:
				updated_product = Product(
					id=product_id,
					**data.model_dump(exclude_unset=True),
					**({"unit": product.unit} if "unit" not in data.model_fields_set else {}),
				)
				products[index] = updated_product
				write_products(products)
				return updated_product
	raise HTTPException(status_code=404, detail="Producto no encontrado")


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: int):
	with inventory_lock:
		products = read_products()
		for index, product in enumerate(products):
			if product.id == product_id:
				products.pop(index)
				write_products(products)
				return Response(status_code=204)
	raise HTTPException(status_code=404, detail="Producto no encontrado")