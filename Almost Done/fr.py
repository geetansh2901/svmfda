from tkinter import * 


window = Tk()
window.title('Smart Vending Machine Software ')

but1 = Button(window, text='Withdraw', width=40, height=10,bg='#7cfc00', fg='black', bd=5)
but1.grid(row=1, column=1)
but2 = Button(window, text='Change Pin', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
but2.grid(row=1, column=3)
but3 = Button(window, text='View Balance', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
but3.grid(row=3, column=1)

# logo = PhotoImage(file='me.jpg')
# w1 = Label(window,image=logo).grid(row=2,column=2)

but4 = Button(window, text='Start', width=40, height=10,
              bg='#7cfc00', fg='black', bd=5)
but4.grid(row=3, column=3)

window.mainloop()