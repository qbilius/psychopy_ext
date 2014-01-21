from psychopy import visual, core
from .. import exp
import unittest


class TestSVG(unittest.TestCase):

    def test_write(self):
        win = visual.Window([600,400], units='height')
        svg = exp.SVG(win, filename='stims')

        circle = visual.Circle(win, pos=(-.5,0), fillColor='yellow', lineColor=None)
        circle.draw()
        svg.write(circle)

        line = visual.Line(win, pos=(0,0), lineColor='black', lineWidth=5)
        line.draw()
        svg.write(line)

        rect = visual.Rect(win, height=.8, pos=(.5,0))
        rect.draw()
        svg.write(rect)

        shape = visual.ShapeStim(win, fillColor='blue', opacity=.5)
        shape.draw()
        svg.write(shape)

        text = visual.TextStim(win, pos=(.5,0.25))
        text.draw()
        svg.write(text)
        
        thick = exp.ThickShapeStim(win, vertices=[(-.5,.5),(.5,-.5)], lineWidth=.01)
        thick.draw()
        svg.write(thick)

        win.flip()
        #win.getMovieFrame()
        #win.saveMovieFrames('stims.png')
        #svg.save()

        core.wait(5)
        #win.close()

if __name__ == '__main__':
    unittest.main()
