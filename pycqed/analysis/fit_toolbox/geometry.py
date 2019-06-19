import numpy as np
import math
import matplotlib.pyplot as plt

def getLine(p1, p2, eps=1e-30):
    """
    p1 is a tuple of the first point
    p2 is a tuple of the second point
    returns a tuple of the slope and y-intercept of the line going throug both points
    """

    if abs(p1[0] - p2[0]) < eps:
        slope = 1 / eps
    else:
        slope = float((p1[1] - p2[1]) / (p1[0] - p2[0]))
    yint = float((-1 * (p1[0])) * slope + p1[1])
    # print(p1,p2)
    # print("line", (slope, yint))
    return (slope, yint)


def getIntersection(line1, line2):
    """
    line1 is a tuple of m and b of the line in the form y=mx+b
    line2 is a tuple of m and b of the line in the form y=mx+b
    returns a tuple of the points of the intersection of the two lines
    """
    slope1, slope2 = line1[0], line2[0]
    yint1, yint2 = line1[1], line2[1]
    matA = np.matrix(str(-1 * slope1) + ' 1;' + str(-1 * slope2) + ' 1')
    matB = np.matrix(str(yint1) + '; ' + str(yint2))
    invA = matA.getI()
    resultant = invA * matB
    return (resultant[0, 0], resultant[1, 0])

def getMidpoint(p1, p2):
    """
    p1 is a tuple of the first point
    p2 is a tuple of the second point
    returns the midpoint, in tuple form, of p1 and p2
    """
    return (((p1[0] + p2[0]) / 2.), ((p1[1] + p2[1]) / 2.))

def perpSlope(slope, eps=1e-30):
    # takes slope(float) and returns the slope of a line perpendicular to it
    # nearly 0 slope, return high slope
    if abs(slope) < eps:
        return 1 / eps
    else:
        return (slope * -1) ** -1


def distance(p1, p2):
    """
    p1 is a tuple of ...
    p2 is a tuple of ...
    returns float of distance between p1 and p2
    """
    return (float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)))


def lineFromSlope(slope, point):
    """
    slope is a float of slope
    point is a tuple of ...
    returns tuple of slope and y intercept
    """
    return (slope, ((slope * (-1 * point[0])) + point[1]))


def angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product / (len1 * len2))


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between - pi and + pi """
    ang = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1],
                                                            a[0] - b[0])
    while ang < - np.pi or ang > np.pi:
        if ang < - np.pi:
            ang += 2 * np.pi
        elif ang > np.pi:
            ang -= 2 * np.pi
    return ang


def rotate(pt, angle):
    pt = np.asarray(pt)
    rot_mx = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return tuple(np.matmul(rot_mx, pt))


def circumcenter(point1, point2, point3, show=False, ax=None):
    mid1 = getMidpoint(point1, point2)
    mid2 = getMidpoint(point2, point3)
    # print(mid1, mid2)
    line1 = getLine(point1, point2)
    line2 = getLine(point2, point3)
    # print(line1, line2)
    perp1 = perpSlope(line1[0])
    perp2 = perpSlope(line2[0])
    # print(perp1, perp2)
    perpbi1 = lineFromSlope(perp1, mid1)
    perpbi2 = lineFromSlope(perp2, mid2)
    circumcent = getIntersection(perpbi1, perpbi2)
    radius = distance(circumcent, point1)
    if show == True:
        xList = [point1[0], point2[0], point3[0], point1[0]]
        yList = [point1[1], point2[1], point3[1], point1[1]]
        if ax is not None:
            ax.plot(xList, yList)
            ax.scatter(circumcent[0], circumcent[1])
        else:
            plt.plot(xList, yList)
            plt.scatter(circumcent[0], circumcent[1])
        plt.show()
    return circumcent, radius