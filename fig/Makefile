.PHONY: help install dev build preview clean export

help:
	@echo "Targets:"
	@echo "  install  Install dependencies"
	@echo "  dev      Run Vite dev server"
	@echo "  build    Build for production"
	@echo "  preview  Preview production build"
	@echo "  clean    Remove build output"
	@echo "  export   Build and export figure to PDF + PNG"

install:
	pnpm install

dev:
	pnpm exec vite

build:
	pnpm run build

preview:
	pnpm exec vite preview

clean:
	rm -rf dist

export: build
	@echo "Starting preview server..."
	@pnpm exec vite preview --port 4200 &
	@sleep 2
	@echo "Exporting PDF..."
	DIALOGMTEB_URL=http://localhost:4200 DIALOGMTEB_PADDING=24 DIALOGMTEB_FORMAT=pdf DIALOGMTEB_OUT=dialogmteb-figure.pdf node scripts/export-pdf.cjs
	@echo "Exporting PNG..."
	DIALOGMTEB_URL=http://localhost:4200 DIALOGMTEB_PADDING=24 DIALOGMTEB_FORMAT=png DIALOGMTEB_OUT=dialogmteb-figure.png node scripts/export-pdf.cjs
	@pkill -f "vite preview --port 4200" 2>/dev/null || true
	@echo "Done: dialogmteb-figure.pdf and dialogmteb-figure.png"
