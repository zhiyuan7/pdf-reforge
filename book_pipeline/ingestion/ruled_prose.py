"""Local coordinates for ruled vertical prose, after its semantic review.

Observed rules only partition a region already confirmed to be prose. They
never determine whether a table is prose. Unruled/ambiguous regions retain the
full-region review. All local images and decisions remain reviewable.
"""
METHOD = 'ruled_prose_geometry_v2'


def partition_cells(rgb):
    """Find near-continuous rules despite slight scan skew; return RTL cells.

    Hough segments alone can follow repeated stems of vertical characters.
    A candidate must have ink along 88% of the entire register, including the
    extrapolated ends, before it can split text. Coordinates are pixel edges.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if min(height, width) < 100:
        return []
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 20)

    def rules(mask, vertical):
        # Transposing gives the same geometry/continuity policy on both axes.
        if not vertical:
            mask = mask.T
        h, w = mask.shape
        lines = cv2.HoughLinesP(mask, 1, np.pi/720, threshold=max(40, round(h*.05)),
                               minLineLength=h*.65, maxLineGap=h*.02)
        selected = []
        for x0, y0, x1, y1 in ([] if lines is None else lines.reshape(-1, 4)):
            if abs(y1-y0) <= .65*h or abs(x1-x0) >= .025*abs(y1-y0):
                continue
            yy = np.arange(h)
            xx = np.rint(x0+(yy-y0)*(x1-x0)/(y1-y0)).astype(int)
            radius = max(2, round(w*.0015))
            coverage = np.asarray([mask[yy, np.clip(xx+d, 0, w-1)] > 0
                                   for d in range(-radius, radius+1)]).any(0).mean()
            if coverage >= .88:
                selected.append(float(xx.mean()))
        groups = []
        for value in sorted(selected):
            if groups and value-groups[-1][-1] <= .025*w:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [round(float(np.mean(g))) for g in groups if .025*w < np.mean(g) < .975*w]

    rows = [0, *rules(ink, False), height]
    cells = []
    for top, bottom in zip(rows, rows[1:]):
        if bottom-top < .12*height:
            return []  # Dense row rules could instead be an unresolved table.
        edges = [0, *rules(ink[top:bottom], True), width]
        for left, right in reversed(list(zip(edges, edges[1:]))):
            if right-left < .025*width:
                return []
            cells.append([left, top, right, bottom])
    # Avoid an uncontrolled number of paid requests or an arbitrary crop.
    return cells if 2 <= len(cells) <= 12 else []


def page_box(local, cell, size):
    """Map local normalized coordinates through the actual integer crop."""
    left, top, right, bottom = cell
    width, height = size
    return [(left+local[0]*(right-left))/width,
            (top+local[1]*(bottom-top))/height,
            (left+local[2]*(right-left))/width,
            (top+local[3]*(bottom-top))/height]
